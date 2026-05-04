# DATUM — Interactive Data Analysis Terminal

**ECE 272C · University of California, Santa Barbara**  
**TA:** Seoyeon Kim

---

## 1. Overview

DATUM is a stateful, interactive data analysis agent built on top of a LangGraph state machine, a FastAPI backend, and a React frontend. Rather than treating each user question independently, the system maintains execution results across turns, enabling follow-up queries, chart re-renders, and filtered views of previous outputs — all without recomputing from the original dataset.

The interaction loop is:

1. The user uploads a CSV and asks a question in natural language
2. The agent classifies the query intent, generates and executes Python/Pandas code, and optionally produces an interactive Plotly chart
3. The execution result is stored as a structured JSON artifact in application state
4. The user can ask follow-up questions that operate directly on that artifact

---

## 2. Core Requirements Met

| Requirement | Implementation |
|---|---|
| Natural language → executable analysis | `code_generation_node` prompts Claude Sonnet to write Pandas code |
| Execution results as persistent state | `execution_result` field in `AgentState` — a JSON-safe list, dict, or scalar |
| Follow-up queries without re-reading CSV | `query_classifier_node` detects intent; `followup` queries operate on `df_prev` |
| Visualization on execution result only | `visualization_node` receives the result artifact, never re-reads the CSV |
| Conversational memory | `chat_history` accumulates LangChain message objects across turns |
| Frontend + backend | React (Vite) + FastAPI with Server-Sent Events streaming |

---

## 3. System Design

The system is built as a **LangGraph state machine** with five sequential nodes, a conditional retry edge, and a shared typed state object.

```
                        ┌─────────────────────────────┐
                        │  (error + retry_count < 2)  │
classify → generate_code → execute → [retry_router] ──┘
                                          │
                                    (success / exhausted)
                                          │
                                      visualize → generate_answer → END
```

### Component Map

| Component | File | Responsibility |
|---|---|---|
| State schema | `nodes.py` · `AgentState` | Typed shared state — all nodes read and write to this |
| Query classifier | `nodes.py` · `query_classifier_node` | Classifies intent into `describe / replot / followup / fresh` |
| Code generation | `nodes.py` · `code_generation_node` | Translates natural language to Pandas code (or prose for describe) |
| Execution sandbox | `nodes.py` · `execution_node` | Runs generated code, serialises result to JSON-safe types |
| Visualization agent | `nodes.py` · `visualization_node` | Decides chart type, generates and executes Plotly code |
| Answer generation | `nodes.py` · `generation_node` | Produces final plain-English response |
| Graph assembly | `agent.py` | Wires nodes and edges, exposes `run_agent()` |
| Backend API | `backend/main.py` | FastAPI — `/api/upload`, `/api/chat/stream` (SSE), `/api/chat` |
| Frontend | `frontend/src/App.jsx` + `App.css` | React chat UI with live streaming status |

---

## 4. Execution Results as System State

Every execution step produces a **structured JSON artifact** stored in `AgentState.execution_result`. This is the central design decision of the system — the result is not just an output but a clean interface between all downstream components.

### Representation

Results are normalised into one of four shapes before storage:

| Shape | When used | Example |
|---|---|---|
| `List[Dict]` | DataFrame result | `[{"Major": "CS", "Count": 14}, ...]` |
| `{"value": scalar}` | Single number / string | `{"value": 42}` |
| `{"empty": True, "columns": [...]}` | Filter returned zero rows | `{"empty": True, "columns": ["Major"]}` |
| `{"error": str}` | Code raised an exception | `{"error": "Python Execution Error: ..."}` |
| `{"describe": str}` | General overview question | `{"describe": "This dataset contains..."}` |
| `{"cannot_answer": str}` | Column doesn't exist | `{"cannot_answer": "No gender column."}` |

### Why JSON-safe normalisation matters

Pandas DataFrames contain `numpy.int64`, `pandas.StringDtype`, and other types that Pydantic cannot serialise. The `_make_json_safe()` helper recursively converts every value to a standard Python primitive before the result leaves `execution_node`, preventing FastAPI 500 errors.

### Why this representation

- **Passable between components** — a plain Python list of dicts can be passed directly to `pd.DataFrame()` in follow-up queries or Plotly chart generation without any deserialisation step
- **Inspectable during debugging** — the backend terminal prints each result shape clearly
- **Reusable across turns** — the React frontend holds `prevResult` in state and sends it with every request; the backend reinjects it into the LangGraph state as `execution_result`

### Trade-offs

The list-of-dicts format loses DataFrame index information and column dtype metadata. For most analytical queries this is irrelevant, but it means operations like `df.resample()` require re-reading the CSV rather than operating on the stored result.

---

## 5. Visualization Agent

The `visualization_node` is a dedicated LangGraph node that operates **exclusively on the execution result** — it never reads the CSV.

### Input state

```python
state["execution_result"]  # list of dicts or scalar dict
state["current_query"]     # user's original question (for chart type decision)
```

### Decision logic

The node first calls the LLM with the result's column names, row count, a 3-row sample, and the user's query. The LLM outputs either `NO` or valid Python code using `plotly.express`. This makes chart-type selection context-aware: a time-series query gets a line chart, a category comparison gets a bar chart, a proportion question gets a pie chart — without any hardcoded rules.

### Execution

Plotly code runs inside a pre-loaded sandbox where `px`, `pd`, `np`, and `json` are already injected as globals. The generated code receives the result DataFrame as `data` and assigns the figure to `fig`. The node extracts `fig.to_json()` and returns the parsed dict.

### Output state

```python
state["visualization_figure"]  # Plotly JSON dict, or None
```

### How the frontend uses it

The React frontend renders charts using `window.Plotly.newPlot()` called directly on a `<div>` ref — bypassing the `react-plotly.js` npm package which uses `eval()` and is blocked by Vite's Content Security Policy.

---

## 6. Frontend — Chatbot UI

Built with React + Vite. Styled as a dark vintage data terminal using `Special Elite` (typewriter) and `Lora` (serif) fonts.

### Features

- **Text input** — multiline textarea, Enter to submit, Shift+Enter for newline
- **Chat history** — scrollable message pane with user bubbles (right) and AI responses (left)
- **Visualization display** — Plotly charts render inline inside the AI message bubble
- **Live streaming status** — replaces a plain spinner with per-node progress labels:
  ```
  ✓ Classifying intent...
  ✓ Writing analysis code...
  ▸ Executing code...  ▌
  ```
- **Loading indicator** — animated dots while streaming, pulsing `▸` caret in input
- **Error handling** — error strip with message when backend returns a non-200 or throws
- **CSV upload** — hidden file input triggered by a styled label button; auto-updates the active dataset path and clears memory

### Query suggestion chips

The empty state shows four clickable example queries covering the main use cases — overviews, row previews, charts, and filters — so first-time users know what the system can do.

---

## 7. Backend

Built with **FastAPI**. Uvicorn serves the app with `--reload` for development.

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/upload` | POST | Accepts a CSV file, saves to `datasets/`, invalidates schema cache, returns `{filename, dataset_path}` |
| `/api/chat/stream` | POST | **Primary endpoint.** Runs LangGraph with `.stream()`, emits SSE events per node |
| `/api/chat` | POST | Non-streaming fallback, returns complete `ChatResponse` JSON |

### Streaming architecture (SSE)

`/api/chat/stream` uses LangGraph's `.stream(stream_mode="updates")` which yields one `{node_name: state_update}` dict as each node completes. The FastAPI `StreamingResponse` emits these as Server-Sent Events with two event types:

```json
{"type": "progress", "message": "⚡ Executing code..."}
{"type": "done", "final_answer": "...", "visualization_figure": {...}, ...}
```

The frontend consumes the stream with the Fetch API's `ReadableStream`, parses each `data:` line, and updates the UI incrementally.

### Structured response

```python
class ChatResponse(BaseModel):
    final_answer: str
    visualization_figure: Optional[Dict[str, Any]]
    execution_result: Optional[Any]
    chat_history: List[Dict[str, str]]
```

---

## 8. Memory

Memory is implemented at two levels: **within a session** via LangGraph state, and **across turns** via the frontend holding and re-sending the execution result artifact.

### What is stored

- `chat_history` — list of `HumanMessage` / `AIMessage` LangChain objects, accumulated by `generation_node` on every turn and sent back to the frontend as `[{"role": "user"|"ai", "content": str}]`
- `execution_result` — the structured JSON artifact from the last successful execution. The React frontend holds this in `useState` as `prevResult` and includes it in every subsequent request body

### How it influences reasoning — the Query Classifier

Before any code is generated, `query_classifier_node` makes an explicit intent decision using a dedicated LLM call. It classifies the query into one of four types:

| Type | Behaviour |
|---|---|
| `describe` | Bypasses pandas entirely — answers from schema metadata |
| `replot` | Passes `prev_result` unchanged to `visualization_node` for a new chart type |
| `followup` | Operates on `df_prev` (previous result as DataFrame); may also load full CSV if needed |
| `fresh` | Ignores `prev_result`, loads CSV fresh |

The `followup` path includes an **escape hatch**: if the follow-up query requires data not present in the filtered result (e.g. "compare these 5 rows against the overall average"), the generated code is permitted to also load the full CSV.

### Self-healing memory

When execution fails, `retry_count` is incremented and the conditional `retry_router` edge in `agent.py` routes back to `code_generation_node` with the exact error message injected into the prompt. This repeats up to `MAX_RETRIES = 2` times before falling through to `generation_node` with an error explanation.

---

## Project Structure

```
project/
├── agent.py              # LangGraph graph assembly and run_agent()
├── nodes.py              # All five nodes + helpers + AgentState schema
├── backend/
│   └── main.py           # FastAPI app — upload, stream, fallback endpoints
├── frontend/
│   ├── index.html        # Vite entry — Plotly CDN script loaded here
│   └── src/
│       ├── App.jsx       # React UI — chat, streaming, PlotlyChart component
│       └── App.css       # Vintage terminal dark theme
├── datasets/             # Uploaded CSVs stored here
└── results.csv           
```

## Setup

```bash
# Backend
pip install fastapi uvicorn langgraph langchain-anthropic pandas plotly python-multipart
export ANTHROPIC_API_KEY=your_key_here
uvicorn backend.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

---

## Failure Case

When a user asks a follow-up question that is semantically ambiguous — for example, "show me more" after receiving a filtered result — the classifier occasionally misclassifies the intent as `fresh` instead of `followup`, causing the agent to reload the CSV and lose the filtered context. This occurs because the classifier's YES/NO prompt has limited signal when the query contains no explicit reference to the previous result. A future improvement would be to include a short excerpt of the previous result directly in the classification prompt, giving the LLM stronger grounding for the decision.