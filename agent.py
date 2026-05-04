from langgraph.graph import StateGraph, END
from nodes import (
    AgentState,
    query_classifier_node,
    code_generation_node,
    execution_node,
    visualization_node,
    generation_node,
)

MAX_RETRIES = 2  # Maximum self-healing attempts before giving up


def build_agent_graph():
    """
    Compiles the LangGraph workflow.

    Graph structure:
                        ┌─────────────────────────────┐
                        │                             │ (retry, retry_count < MAX)
    classify ──► generate_code ──► execute ──► [router]
                                                  │
                                          (success OR retries exhausted)
                                                  │
                                              visualize ──► generate_answer ──► END
    """
    workflow = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────

    workflow.add_node("classify", query_classifier_node)
    workflow.add_node("generate_code", code_generation_node)

    def execute_wrapper(state):
        return execution_node(state, state.get("generated_code", ""))
    workflow.add_node("execute", execute_wrapper)

    workflow.add_node("visualize", visualization_node)
    workflow.add_node("generate_answer", generation_node)

    # ── Edges ─────────────────────────────────────────────────────────────────

    # classify always flows into generate_code
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "generate_code")
    workflow.add_edge("generate_code", "execute")

    # GAP 2 FIX — conditional retry router after execution
    def retry_router(state: AgentState) -> str:
        """
        Routes to:
          - "generate_code"  if execution failed AND we haven't hit MAX_RETRIES
          - "visualize"      otherwise (success, empty, cannot_answer, retries exhausted)
        """
        result = state.get("execution_result", {})
        retry_count = state.get("retry_count", 0)

        is_error = isinstance(result, dict) and "error" in result
        # cannot_answer and empty are valid terminal states — don't retry them
        is_terminal = isinstance(result, dict) and (
            "cannot_answer" in result or "empty" in result
        )

        if is_error and not is_terminal and retry_count < MAX_RETRIES:
            print(
                f"\n--- RETRY ROUTER: error detected, attempt "
                f"{retry_count}/{MAX_RETRIES} — routing back to generate_code ---\n"
            )
            return "generate_code"

        return "visualize"

    workflow.add_conditional_edges(
        "execute",
        retry_router,
        {
            "generate_code": "generate_code",
            "visualize": "visualize",
        }
    )

    workflow.add_edge("visualize", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# Compile once at import time
agent_executor = build_agent_graph()


def run_agent(
    query: str,
    dataset_path: str,
    chat_history: list = None,
    prev_result: dict = None,
):
    """Entry point called by the FastAPI backend."""
    initial_state = {
        "current_query": query,
        "dataset_path": dataset_path,
        "chat_history": chat_history or [],
        "execution_result": prev_result,
        "visualization_figure": None,
        "query_type": None,
        "retry_count": 0,
    }

    final_state = agent_executor.invoke(initial_state)
    return final_state
