import json
import functools
import numpy as np
import pandas as pd
import plotly.express as px
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

# Initialize Claude
llm = ChatAnthropic(
    model_name="claude-sonnet-4-6",
    temperature=0,
    max_tokens=4096
)

# ── Pre-load exec sandbox globals (Improvement 3) ─────────────────────────────
# Injecting libraries here means Claude's generated code never needs to import
# them — shorter code, no import latency, and we control the environment.
EXEC_GLOBALS = {
    "pd": pd,
    "json": json,
    "np": np,
}
VIZ_GLOBALS = {
    "pd": pd,
    "json": json,
    "np": np,
    "px": px,
}

# ── State Schema ──────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    chat_history: List[Any]
    current_query: str
    dataset_path: str
    generated_code: Optional[str]
    execution_result: Optional[Any]
    visualization_code: Optional[str]
    visualization_figure: Optional[Dict[str, Any]]
    final_answer: str
    query_type: Optional[str]   # "describe"|"replot"|"followup"|"fresh"
    retry_count: Optional[int]


# ── Helpers ───────────────────────────────────────────────────────────────────

# Improvement 1: lru_cache — schema is read once per unique file path and
# remembered in memory. Drops I/O on every subsequent message to zero.
@functools.lru_cache(maxsize=10)
def _get_schema_hint(dataset_path: str) -> str:
    """
    Reads CSV header, types, sample values, and 3 real rows.
    Cached per file path — subsequent calls are instant.
    """
    try:
        df_sample = pd.read_csv(dataset_path, nrows=100)
        cols = df_sample.columns.tolist()
        dtypes = df_sample.dtypes.to_dict()

        value_hints = []
        for col in cols:
            if dtypes[col] == object:
                unique_vals = df_sample[col].dropna().unique().tolist()[:8]
                value_hints.append(f"  - '{col}' (string): sample values = {unique_vals}")
            else:
                value_hints.append(
                    f"  - '{col}' ({dtypes[col]}): "
                    f"min={df_sample[col].min()}, max={df_sample[col].max()}"
                )

        sample_rows = df_sample.head(3).to_dict(orient="records")

        return (
            f"\nCRITICAL SCHEMA — the dataset has ONLY these {len(cols)} columns:\n"
            + "\n".join(value_hints)
            + f"\n\nSample rows (first 3):\n{sample_rows}"
            + "\n\nCRITICAL RULES:"
            + "\n1. Use ONLY the column names listed above. Do NOT invent or assume any other columns."
            + "\n2. If the question CANNOT be answered from these columns, output ONLY: "
            "CANNOT_ANSWER: <one sentence explaining what is missing>"
            "\n3. When filtering strings, ALWAYS use .str.lower().str.strip() on both sides."
            "\n4. If a filter returns zero rows, assign the empty DataFrame to `result` — do not raise."
            "\n5. Libraries pd, np, json are already available — do NOT import them."
        )
    except Exception as e:
        print(f"Schema hint error: {e}")
        return ""


def _invalidate_schema_cache(dataset_path: str):
    """Call this when a new file is uploaded so the cache refreshes."""
    _get_schema_hint.cache_clear()


def _is_empty_result(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, list) and len(result) == 0:
        return True
    if isinstance(result, dict):
        if "error" in result or "cannot_answer" in result:
            return False
        val = result.get("value")
        if val is None:
            return True
        if isinstance(val, (list, dict)) and len(val) == 0:
            return True
    return False


def _safe_build_dataframe(result: Any) -> Optional[pd.DataFrame]:
    try:
        if isinstance(result, list):
            return pd.DataFrame(result)
        elif isinstance(result, dict):
            if "value" in result or "cannot_answer" in result:
                return None
            return pd.DataFrame([result])
        return None
    except Exception as e:
        print(f"DataFrame construction error: {e}")
        return None


def _prev_result_summary(prev_result: Any) -> str:
    if isinstance(prev_result, list):
        cols = list(prev_result[0].keys()) if prev_result else []
        return (
            f"list of {len(prev_result)} records, "
            f"columns: {cols}, "
            f"first row: {prev_result[0] if prev_result else 'N/A'}"
        )
    if isinstance(prev_result, dict):
        return str(prev_result)[:300]
    return str(prev_result)[:300]


def _make_json_safe(obj: Any) -> Any:
    """Recursively converts anything into JSON-serializable Python primitives."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(i) for i in obj]
    if isinstance(obj, pd.DataFrame):
        return json.loads(obj.to_json(orient="records", date_format="iso"))
    if isinstance(obj, pd.Series):
        return json.loads(obj.to_json(date_format="iso"))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if 'Dtype' in type(obj).__name__:
        return str(obj)
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


# ── Node 0: Query Classifier ──────────────────────────────────────────────────

def query_classifier_node(state: AgentState) -> dict:
    """
    Classifies each query into one of four types:

      - "describe"  : general overview/explanation — no pandas needed
      - "replot"    : different chart of same data
      - "followup"  : operate on previous result (with CSV escape hatch)
      - "fresh"     : new question from raw CSV
    """
    query = state["current_query"]
    prev_result = state.get("execution_result")

    # Always check describe-intent first
    desc_response = llm.invoke([HumanMessage(content=(
        f'User query: "{query}"\n\n'
        "Is this a general/descriptive question about what the dataset IS — "
        "asking for an overview, summary, explanation of columns, or what the file contains? "
        "Examples: 'what does this file contain', 'give me an overview', "
        "'what can you tell me about this data', 'describe the dataset', 'what columns are there'\n\n"
        "Output ONLY one word: YES or NO."
    ))])
    if "YES" in desc_response.content.strip().upper():
        print(f"\n--- CLASSIFIER: describe | query='{query}' ---\n")
        return {"query_type": "describe", "retry_count": 0}

    # No usable previous result → fresh
    if not prev_result or _is_empty_result(prev_result) or (
        isinstance(prev_result, dict) and ("error" in prev_result or "cannot_answer" in prev_result)
    ):
        print(f"\n--- CLASSIFIER: fresh (no usable prev_result) | query='{query}' ---\n")
        return {"query_type": "fresh", "retry_count": 0}

    prev_summary = _prev_result_summary(prev_result)

    response = llm.invoke([HumanMessage(content=(
        f'Previous result summary: {prev_summary}\n\n'
        f'New user query: "{query}"\n\n'
        "Classify this query into exactly one of these three categories:\n"
        "  REPLOT   — only wants a different chart of the exact same data\n"
        "  FOLLOWUP — wants to compute, filter, sort, or transform the previous result\n"
        "  FRESH    — entirely new question requiring the raw dataset\n\n"
        "Output ONLY one word: REPLOT, FOLLOWUP, or FRESH. No explanation."
    ))])
    decision = response.content.strip().upper().rstrip(".")

    if "REPLOT" in decision:
        query_type = "replot"
    elif "FOLLOWUP" in decision:
        query_type = "followup"
    else:
        query_type = "fresh"

    print(f"\n--- CLASSIFIER: {query_type} | query='{query}' ---\n")
    return {"query_type": query_type, "retry_count": 0}


# ── Node 1: Code Generation ───────────────────────────────────────────────────

def code_generation_node(state: AgentState) -> dict:
    """
    Generates pandas code appropriate to the query_type:
      - describe:  answers in prose from schema, no code needed
      - replot:    passthrough — sends prev_result to visualization node unchanged
      - followup:  operates on prev_result; can also load CSV if needed (Improvement 2)
      - fresh:     loads from CSV
    """
    query = state["current_query"]
    prev_result = state.get("execution_result")
    query_type = state.get("query_type", "fresh")
    retry_count = state.get("retry_count", 0)
    schema_hint = _get_schema_hint(state["dataset_path"])

    # ── Describe short-circuit ────────────────────────────────────────────────
    if query_type == "describe":
        print("\n--- CODE GEN: describe — prose from schema ---\n")
        response = llm.invoke([HumanMessage(content=(
            f'The user asked: "{query}"\n\n'
            f'Dataset info:\n{schema_hint}\n\n'
            "Write a clear, friendly overview of what this dataset contains. "
            "Cover: number of columns, what each column represents, data types, "
            "and 2-3 observations about what kinds of questions this data could answer. "
            "Use plain English — no code, no markdown headers. "
            "Bold column names."
        ))])
        prose = response.content.strip()
        print(f"\n--- DESCRIBE ANSWER ---\n{prose[:200]}\n-----------------------\n")
        # Remove final_answer from here — let generation_node produce it
        # so the LangGraph state machine works correctly in .stream() mode.
        return {
            "generated_code": "DESCRIBE_COMPLETE",
            "execution_result": {"describe": prose},
        }

    # ── Replot short-circuit ──────────────────────────────────────────────────
    if query_type == "replot" and prev_result and not _is_empty_result(prev_result):
        print("\n--- CODE GEN: replot — passing prev_result unchanged ---\n")
        if isinstance(prev_result, list):
            passthrough_code = f"result = pd.DataFrame({json.dumps(prev_result)})"
        elif isinstance(prev_result, dict) and "value" in prev_result:
            passthrough_code = f"result = {repr(prev_result['value'])}"
        else:
            passthrough_code = f"result = {repr(prev_result)}"
        return {"generated_code": passthrough_code}

    system_prompt = (
        "You are a Python data analysis agent.\n"
        "Write Python code using pandas to answer the user's question.\n"
        "CRITICAL: Assign the final result to a variable named exactly `result`.\n"
        "CRITICAL: Output ONLY valid Python code — no prose, no markdown fences.\n"
        "CRITICAL: Do NOT import matplotlib, seaborn, or plotly.\n"
        "CRITICAL: pd, np, and json are already available — do NOT import them.\n"
        "CRITICAL: When filtering strings, ALWAYS use .str.lower().str.strip() on both sides.\n"
        "CRITICAL: If a filter returns zero rows, assign the empty DataFrame to `result`.\n"
        "CRITICAL: If the question cannot be answered from available data, output ONLY:\n"
        "CANNOT_ANSWER: <one sentence explaining what is missing>"
    )

    # ── Improvement 2: followup with CSV escape hatch ─────────────────────────
    # Claude now has access to BOTH the filtered result AND the full dataset.
    # This fixes queries like "compare these 5 students against the overall average"
    # which need the full CSV even though they're semantically a follow-up.
    if query_type == "followup" and prev_result:
        if isinstance(prev_result, list):
            prev_repr = f"pd.DataFrame({json.dumps(prev_result)})"
        elif isinstance(prev_result, dict) and "value" in prev_result:
            prev_repr = repr(prev_result["value"])
        else:
            prev_repr = repr(prev_result)

        context = (
            f"MEMORY MODE — this is a FOLLOW-UP query.\n\n"
            f"The previous result is loaded as `df_prev` (already a DataFrame):\n"
            f"df_prev = {prev_repr}\n\n"
            f"It contains: {_prev_result_summary(prev_result)}\n\n"
            f"Operate on `df_prev` to answer the query. "
            f"HOWEVER — if you need data from the original dataset that is not in `df_prev` "
            f"(e.g. an overall average, a column not present in the filtered result), "
            f"you MAY also load: df = pd.read_csv('{state['dataset_path']}')\n\n"
            f"Always assign your final answer to `result`.\n"
            f"{schema_hint}"
        )

    elif retry_count > 0 and isinstance(prev_result, dict) and "error" in prev_result:
        context = (
            f"RETRY ATTEMPT {retry_count}: your previous code failed:\n"
            f"{prev_result['error']}\n\n"
            f"Reload from scratch: df = pd.read_csv('{state['dataset_path']}')\n"
            f"Fix the specific error above. Do NOT repeat the same mistake.\n"
            f"{schema_hint}"
        )

    else:
        context = (
            f"Load the dataset: df = pd.read_csv('{state['dataset_path']}')\n"
            f"{schema_hint}"
        )

    messages = [
        SystemMessage(content=f"{system_prompt}\n\n{context}"),
        HumanMessage(content=query)
    ]
    response = llm.invoke(messages)
    raw_code = response.content.strip()

    # CANNOT_ANSWER detection (before and after fence stripping)
    if raw_code.startswith("CANNOT_ANSWER:"):
        return {"generated_code": raw_code}
    if "```python" in raw_code:
        raw_code = raw_code.split("```python")[1].split("```")[0].strip()
    elif "```" in raw_code:
        raw_code = raw_code.split("```")[1].strip()
    if raw_code.startswith("CANNOT_ANSWER:"):
        return {"generated_code": raw_code}

    print(f"\n--- GENERATED CODE (type={query_type}, retry={retry_count}) ---")
    print(raw_code)
    print("--------------------------------------------------------------\n")

    return {"generated_code": raw_code}


# ── Node 2: Execution ─────────────────────────────────────────────────────────

def execution_node(state: AgentState, generated_code: str) -> dict:
    """
    Executes generated code in a pre-loaded sandbox (Improvement 3).
    pd, np, json are pre-injected — Claude's code doesn't need to import them.
    """
    # DESCRIBE_COMPLETE — prose already written, nothing to execute
    if generated_code == "DESCRIBE_COMPLETE":
        return {}

    # CANNOT_ANSWER short-circuit
    if generated_code.startswith("CANNOT_ANSWER:"):
        reason = generated_code[len("CANNOT_ANSWER:"):].strip()
        return {
            "execution_result": {"cannot_answer": reason},
            "final_answer": reason
        }

    # Build local scope — inject prev_result if useful
    local_vars = {}
    prev = state.get("execution_result")
    if prev and not _is_empty_result(prev) and not (isinstance(prev, dict) and "error" in prev):
        local_vars["prev_result"] = prev
        # Also expose as df_prev for followup queries
        if isinstance(prev, list):
            local_vars["df_prev"] = pd.DataFrame(prev)

    try:
        # Improvement 3: use EXEC_GLOBALS instead of globals()
        # pd, np, json are pre-loaded — no import needed in generated code
        exec(generated_code, {**EXEC_GLOBALS}, local_vars)  # noqa: S102
        raw_result = local_vars.get("result")

        if isinstance(raw_result, pd.DataFrame):
            if raw_result.empty:
                return {
                    "execution_result": {"empty": True, "columns": raw_result.columns.tolist()},
                    "final_answer": "Query returned no rows."
                }
            structured_result = json.loads(raw_result.to_json(orient="records", date_format="iso"))

        elif isinstance(raw_result, pd.Series):
            if raw_result.empty:
                return {"execution_result": {"empty": True}, "final_answer": "Query returned no data."}
            structured_result = json.loads(raw_result.to_json(date_format="iso"))

        else:
            try:
                structured_result = {"value": raw_result.item()}
            except AttributeError:
                if isinstance(raw_result, dict):
                    structured_result = _make_json_safe(raw_result)
                elif isinstance(raw_result, (int, float, str, bool, list, type(None))):
                    structured_result = {"value": raw_result}
                else:
                    structured_result = {"value": str(raw_result)}

        return {"execution_result": structured_result, "final_answer": "Execution successful."}

    except Exception as e:
        error_msg = f"Python Execution Error: {str(e)}"
        print(f"\n--- EXECUTION ERROR ---\n{error_msg}\n-----------------------\n")
        return {
            "execution_result": {"error": error_msg},
            "final_answer": "Execution failed.",
            "retry_count": state.get("retry_count", 0) + 1
        }


# ── Node 3: Visualization ─────────────────────────────────────────────────────

def visualization_node(state: AgentState) -> dict:
    """
    Generates Plotly code using pre-injected VIZ_GLOBALS (Improvement 3).
    px is available in scope — Claude's code doesn't need to import it.
    """
    result = state.get("execution_result")
    query = state["current_query"]

    if not result:
        return {"visualization_figure": None}
    if isinstance(result, dict) and (
        "error" in result or "empty" in result
        or "cannot_answer" in result or "describe" in result
    ):
        return {"visualization_figure": None}
    if isinstance(result, dict) and "value" in result:
        return {"visualization_figure": None}

    df = _safe_build_dataframe(result)
    if df is None or df.empty:
        return {"visualization_figure": None}

    response = llm.invoke([HumanMessage(content=(
        f'User query: "{query}"\n'
        f"Data columns: {df.columns.tolist()}\n"
        f"Row count: {len(df)}\n"
        f"Sample (first 3 rows): {df.head(3).to_dict(orient='records')}\n\n"
        "Should a chart be shown? Useful for comparisons, distributions, trends.\n"
        "If YES: write ONLY valid Python using plotly.express. "
        "`px` and `data` (the DataFrame) are already in scope — do NOT import anything or reload CSV. "
        "Assign the figure to `fig`.\n"
        "If NO: output exactly NO.\n"
        "CRITICAL: Output ONLY code or the word NO."
    ))])
    decision = response.content.strip()

    if "```python" in decision:
        decision = decision.split("```python")[1].split("```")[0].strip()
    elif "```" in decision:
        decision = decision.split("```")[1].strip()

    if decision.upper().rstrip(".'\"") == "NO":
        return {"visualization_figure": None}

    local_vars = {"data": df}
    try:
        print(f"\n--- PLOTLY CODE ---\n{decision}\n-------------------\n")
        exec(decision, {**VIZ_GLOBALS}, local_vars)  # noqa: S102
        fig = local_vars.get("fig")
        if fig:
            return {"visualization_figure": json.loads(fig.to_json())}
    except Exception as e:
        print(f"Visualization error: {e}")

    return {"visualization_figure": None}


# ── Node 4: Answer Generation ─────────────────────────────────────────────────

def generation_node(state: AgentState) -> dict:
    """Generates the final conversational answer for every result type."""
    query = state["current_query"]
    result = state.get("execution_result")
    fig = state.get("visualization_figure")

    if isinstance(result, dict) and "describe" in result:
        final_text = result["describe"]

    elif isinstance(result, dict) and "cannot_answer" in result:
        final_text = (
            f"I can't answer that with the current dataset. {result['cannot_answer']}\n\n"
            "If you need this analysis, you'd need a dataset that includes the relevant data."
        )

    elif isinstance(result, dict) and "error" in result:
        final_text = (
            f"I ran into an error while analysing the data:\n\n"
            f"> {result['error']}\n\n"
            "Could you rephrase your question or check the dataset is loaded correctly?"
        )

    elif _is_empty_result(result) or (isinstance(result, dict) and "empty" in result):
        cols = result.get("columns", []) if isinstance(result, dict) else []
        col_hint = f" (available columns: {cols})" if cols else ""
        final_text = (
            f"The query ran successfully but returned **no matching rows**{col_hint}. "
            "This is usually a filtering issue — the value you specified might be spelled "
            "or capitalised differently in the dataset."
        )

    else:
        display_result = result
        if isinstance(result, list) and len(result) > 5:
            display_result = f"{result[:5]} ... [{len(result) - 5} more rows omitted]"

        prompt = f"Data result: {display_result}\nUser query: '{query}'\n\n"
        if fig:
            prompt += (
                "A chart has been generated. Give a 1-2 sentence plain-English summary "
                "of what the data shows and tell the user to refer to the chart below. "
                "Do NOT write any code."
            )
        else:
            prompt += (
                "Answer the user's question in plain English using the data above. "
                "Be concise and direct. Do NOT leave the response blank."
            )

        response = llm.invoke([HumanMessage(content=prompt)])
        final_text = response.content.strip() or "Here is the data based on your query."

    print(f"\n--- FINAL ANSWER ---\n{final_text[:200]}...\n--------------------\n")

    new_history = state.get("chat_history", []) + [
        HumanMessage(content=query),
        AIMessage(content=final_text)
    ]
    return {"final_answer": final_text, "chat_history": new_history}
