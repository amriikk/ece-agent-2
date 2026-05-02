# project/backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# Ensure the parent directory is in the path so we can import our agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import run_agent
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="Interactive Data Analysis Agent")

# Allow the React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to the frontend's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected request payload from the frontend
class ChatRequest(BaseModel):
    query: str
    dataset_path: str
    chat_history: Optional[List[Dict[str, str]]] = []
    prev_result: Optional[Dict[str, Any]] = None

# Define the structured response to send back to the frontend
class ChatResponse(BaseModel):
    final_answer: str
    visualization_figure: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None
    chat_history: List[Dict[str, str]]

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert frontend chat history format (dict) to LangChain message objects
        langchain_history = []
        for msg in request.chat_history:
            if msg["role"] == "user":
                langchain_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "ai":
                langchain_history.append(AIMessage(content=msg["content"]))

        # Execute the LangGraph agent
        final_state = run_agent(
            query=request.query,
            dataset_path=request.dataset_path,
            chat_history=langchain_history,
            prev_result=request.prev_result
        )
        
        # Format the updated chat history to send back to the frontend
        formatted_history = []
        for msg in final_state.get("chat_history", []):
            role = "user" if msg.type == "human" else "ai"
            formatted_history.append({"role": role, "content": msg.content})

        return ChatResponse(
            final_answer=final_state.get("final_answer", ""),
            visualization_figure=final_state.get("visualization_figure"),
            execution_result=final_state.get("execution_result"),
            chat_history=formatted_history
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the server:
# uvicorn backend.main:app --reload