from langgraph.graph import StateGraph, END
from nodes import (
    AgentState, 
    code_generation_node, 
    execution_node, 
    visualization_node, 
    generation_node
)

def build_agent_graph():
    """Compiles the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("generate_code", code_generation_node)
    
    # We use a wrapper for execution to pass the generated code from the state/kwargs cleanly
    def execute_wrapper(state):
        return execution_node(state, state.get("generated_code", ""))
    workflow.add_node("execute", execute_wrapper)
    
    workflow.add_node("visualize", visualization_node)
    workflow.add_node("generate_answer", generation_node)

    # Define the execution flow
    workflow.set_entry_point("generate_code")
    workflow.add_edge("generate_code", "execute")
    workflow.add_edge("execute", "visualize")
    workflow.add_edge("visualize", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Compile the graph
    app = workflow.compile()
    return app

# Initialize the agent graph
agent_executor = build_agent_graph()

def run_agent(query: str, dataset_path: str, chat_history: list = None, prev_result: dict = None):
    """Helper function to invoke the agent from the FastAPI backend."""
    initial_state = {
        "current_query": query,
        "dataset_path": dataset_path,
        "chat_history": chat_history or [],
        "execution_result": prev_result,  # Passing in previous state for memory
        "visualization_figure": None
    }
    
    final_state = agent_executor.invoke(initial_state)
    return final_state