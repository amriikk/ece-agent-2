from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import shutil, sys, os, json, asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import agent_executor
from nodes import _invalidate_schema_cache
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Datum — Data Analysis Terminal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Human-readable labels for each LangGraph node (Improvement 4) ────────────
NODE_LABELS = {
    "classify":        "⚙ Classifying intent...",
    "generate_code":   "✍ Writing analysis code...",
    "execute":         "⚡ Executing code...",
    "visualize":       "📊 Generating chart...",
    "generate_answer": "💬 Composing answer...",
}

class ChatRequest(BaseModel):
    query: str
    dataset_path: str
    chat_history: Optional[List[Dict[str, str]]] = []
    prev_result: Optional[Any] = None

class ChatResponse(BaseModel):
    final_answer: str
    visualization_figure: Optional[Dict[str, Any]] = None
    execution_result: Optional[Any] = None
    chat_history: List[Dict[str, str]]

# ── Upload endpoint ────────────────────────────────────────────────────────────
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs("datasets", exist_ok=True)
        file_path = f"datasets/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Invalidate schema cache so next query reads the new file fresh
        _invalidate_schema_cache(file_path)
        return {"filename": file.filename, "dataset_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Streaming chat endpoint (Improvement 4) ───────────────────────────────────
# Uses LangGraph's .stream() with stream_mode="updates" to emit one SSE event
# per node as it completes. The React frontend shows live status messages
# instead of a plain spinner, masking the 10-15s latency completely.
@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    langchain_history = []
    for msg in (request.chat_history or []):
        if msg["role"] == "user":
            langchain_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            langchain_history.append(AIMessage(content=msg["content"]))

    initial_state = {
        "current_query": request.query,
        "dataset_path": request.dataset_path,
        "chat_history": langchain_history,
        "execution_result": request.prev_result,
        "visualization_figure": None,
        "query_type": None,
        "retry_count": 0,
    }

    async def event_generator():
        final_state = {}
        retry_count = 0

        try:
            # stream() yields {node_name: state_update} dicts as each node finishes
            for chunk in agent_executor.stream(initial_state, stream_mode="updates"):
                for node_name, state_update in chunk.items():
                    # Guard: some nodes (e.g. describe short-circuit) return None
                    if state_update is None:
                        state_update = {}
                    final_state.update(state_update)

                    # Track retries for UI feedback
                    new_retry = state_update.get("retry_count", retry_count)
                    if new_retry > retry_count:
                        retry_count = new_retry
                        label = f"🔄 Execution failed — self-healing (attempt {retry_count})..."
                    else:
                        label = NODE_LABELS.get(node_name, f"⚙ {node_name}...")

                    # Emit a "progress" SSE event
                    progress = json.dumps({"type": "progress", "message": label})
                    yield f"data: {progress}\n\n"
                    await asyncio.sleep(0)   # yield control to event loop

            # Emit the final "done" event with the complete result
            formatted_history = []
            for msg in final_state.get("chat_history", []):
                role = "user" if msg.type == "human" else "ai"
                formatted_history.append({"role": role, "content": msg.content})

            result_payload = json.dumps({
                "type": "done",
                "final_answer": final_state.get("final_answer", ""),
                "visualization_figure": final_state.get("visualization_figure"),
                "execution_result": final_state.get("execution_result"),
                "chat_history": formatted_history,
            })
            yield f"data: {result_payload}\n\n"

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_payload = json.dumps({"type": "error", "message": str(e)})
            yield f"data: {error_payload}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )

# ── Non-streaming fallback (kept for compatibility) ───────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        langchain_history = []
        for msg in (request.chat_history or []):
            if msg["role"] == "user":
                langchain_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                langchain_history.append(AIMessage(content=msg["content"]))

        from agent import run_agent
        final_state = run_agent(
            query=request.query,
            dataset_path=request.dataset_path,
            chat_history=langchain_history,
            prev_result=request.prev_result,
        )

        formatted_history = []
        for msg in final_state.get("chat_history", []):
            role = "user" if msg.type == "human" else "ai"
            formatted_history.append({"role": role, "content": msg.content})

        return ChatResponse(
            final_answer=final_state.get("final_answer", ""),
            visualization_figure=final_state.get("visualization_figure"),
            execution_result=final_state.get("execution_result"),
            chat_history=formatted_history,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
