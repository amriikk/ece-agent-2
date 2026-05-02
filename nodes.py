import pandas as pd
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
import os

# Initialize Claude 3.5 Sonnet
llm = ChatAnthropic(
    model_name="claude-3-5-sonnet-20240620", # or the latest sonnet version string
    temperature=0,
    max_tokens=4096
)

# Define the State Schema
class AgentState(TypedDict):
    chat_history: List[Any]
    current_query: str
    dataset_path: str
    execution_result: Optional[Dict[str, Any]]  # The structured state artifact
    visualization_code: Optional[str]
    visualization_figure: Optional[Dict[str, Any]] # Plotly JSON
    final_answer: str


def code_generation_node(state: AgentState) -> dict:
    """Translates natural language to Python/Pandas code."""
    query = state["current_query"]
    prev_result = state.get("execution_result")
    
    # Prompting strategy: If we have a previous result, operate on that. Otherwise, load the dataset.
    system_prompt = """You are a Python data analysis agent. 
    Write Python code using pandas to answer the user's question.
    Assign the final resulting DataFrame or value to a variable named `result`.
    Do NOT include markdown formatting (like ```python) in your output, just the raw code."""
    
    if prev_result:
        context = f"Operate on this previous execution result (available as a dictionary named `prev_result`):\n{prev_result}"
    else:
        context = f"Load the dataset from `{state['dataset_path']}` into a pandas DataFrame."

    messages = [SystemMessage(content=f"{system_prompt}\n{context}"), HumanMessage(content=query)]
    response = llm.invoke(messages)
    
    return {"generated_code": response.content}

def execution_node(state: AgentState, generated_code: str) -> dict:
    """Executes the code and updates the persistent structured result."""
    local_vars = {}
    
    # Inject prev_result into the local namespace if it exists
    if state.get("execution_result"):
        local_vars["prev_result"] = state["execution_result"]
        
    try:
        # In a real production system, I'd wrap this in a secure sandbox.
        exec(generated_code, globals(), local_vars)
        raw_result = local_vars.get("result")
        
        # Convert the result to a structured format (e.g., dictionary)
        if isinstance(raw_result, pd.DataFrame):
            structured_result = raw_result.to_dict(orient="records")
        elif isinstance(raw_result, pd.Series):
            structured_result = raw_result.to_dict()
        else:
            structured_result = {"value": raw_result}
            
        return {"execution_result": structured_result, "final_answer": "Execution successful."}
    except Exception as e:
        return {"final_answer": f"Error executing code: {str(e)}"}

def visualization_node(state: AgentState) -> dict:
    """Decides if visualization is needed and generates Plotly code."""
    result = state.get("execution_result")
    query = state["current_query"]
    
    if not result:
        return {"visualization_figure": None}

    # Use LLM to decide if visualization is useful and generate Plotly code 
    decision_prompt = f"""
    Given the user query: "{query}" and the current data: {result},
    decide if a visualization is useful (e.g., for comparisons, trends, distributions).
    If YES, write ONLY valid Python code using plotly.express or plotly.graph_objects.
    The code must assign the Plotly figure to a variable named `fig`.
    The data is available as a list of dicts/dict named `data`.
    If NO, output the exact word 'NO'.
    """
    
    response = llm.invoke([HumanMessage(content=decision_prompt)])
    decision = response.content.strip()
    
    if decision == "NO":
        return {"visualization_figure": None}
    
    # Execute the generated Plotly code to get the JSON representation
    local_vars = {"data": result}
    try:
        exec(decision, globals(), local_vars)
        fig = local_vars.get("fig")
        # Return the Plotly figure as a JSON-serializable dictionary for the frontend
        return {"visualization_figure": fig.to_dict() if fig else None}
    except Exception as e:
        print(f"Visualization error: {e}") # Log error, but don't crash the agent
        return {"visualization_figure": None}
        
def generation_node(state: AgentState) -> dict:
    """Generates the final natural language answer to display to the user."""
    query = state["current_query"]
    result = state.get("execution_result")
    
    prompt = f"Based on the execution result: {result}, answer the user's query: '{query}' concisely."
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Append to chat history
    new_history = state.get("chat_history", []) + [
        HumanMessage(content=query),
        AIMessage(content=response.content)
    ]
    
    return {"final_answer": response.content, "chat_history": new_history}