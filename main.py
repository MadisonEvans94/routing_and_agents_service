import logging
from langgraph.graph import StateGraph, START, END
from agents.rag_agent import RAGAgent
from agents.final_model_agent import FinalModelAgent
from typing import Dict, Any
from dotenv import load_dotenv  # Import dotenv to load environment variables
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state structure for the graph


class GraphState(Dict[str, Any]):
    query: str
    retrieved_context: str
    final_answer: str

# Node for routing the query


def routing_agent_node(state: GraphState):
    state["retrieved_context"] = routing_agent.route_query(state["query"])
    return state

# Node for determining which model to use based on query complexity


def final_model_agent_node(state: GraphState):
    state["final_answer"] = final_model_agent.route_model(
        state["query"], state["retrieved_context"]
    )
    return state


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize the agents globally
    global routing_agent, final_model_agent
    routing_agent = RAGAgent(openai_api_key)
    final_model_agent = FinalModelAgent()

    # Initialize LangGraph StateGraph
    workflow = StateGraph(GraphState)

    # Add nodes to the workflow
    workflow.add_node("RoutingAgent", routing_agent_node)
    workflow.add_node("FinalModelAgent", final_model_agent_node)

    # Add edges between nodes
    workflow.add_edge(START, "RoutingAgent")
    workflow.add_edge("RoutingAgent", "FinalModelAgent")
    workflow.add_edge("FinalModelAgent", END)

    # Compile and run the graph
    graph_app = workflow.compile()

    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            logger.info("Agent terminated by user.")
            break

        # Run the graph with the user query
        result = graph_app.invoke({"query": query})
        print("\n\n\n------------------------\n\nFinal Answer:\n", result["final_answer"])


if __name__ == "__main__":
    main()
