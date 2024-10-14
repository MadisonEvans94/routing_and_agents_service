import logging
import click
from langgraph.graph import StateGraph, START, END
from agents.rag_agent import RAGAgent
from agents.model_routing_agent import ModelRoutingAgent
from typing import Dict, Any
from dotenv import load_dotenv
import os
from PIL import Image  # For opening the saved image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the state structure for the graph


class GraphState(Dict[str, Any]):
    query: str
    retrieved_context: str
    final_answer: str

# Node for RAG (Retrieval Augmented Generation) query routing


def rag_agent_node(state: GraphState, rag_agent: RAGAgent):
    state["retrieved_context"] = rag_agent.route_query(state["query"])
    return state

# Node for determining which model to use based on query complexity


def model_routing_agent_node(state: GraphState, model_routing_agent: ModelRoutingAgent):
    state["final_answer"] = model_routing_agent.route_model(
        state["query"], state["retrieved_context"]
    )
    return state

# Use click to add command-line arguments


@click.command()
@click.option('--plot', is_flag=True, help="Flag to plot the graph visualization.")
def main(plot):
    # Load environment variables
    load_dotenv()

    # Retrieve the OpenAI API key from environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize the agents
    rag_agent = RAGAgent(openai_api_key)
    model_routing_agent = ModelRoutingAgent()

    # Initialize LangGraph StateGraph
    workflow = StateGraph(GraphState)

    # Add nodes to the workflow and pass the agents
    workflow.add_node(
        "RAGAgent", lambda state: rag_agent_node(state, rag_agent))
    workflow.add_node("ModelRoutingAgent", lambda state: model_routing_agent_node(
        state, model_routing_agent))

    # Add edges between nodes
    workflow.add_edge(START, "RAGAgent")
    workflow.add_edge("RAGAgent", "ModelRoutingAgent")
    workflow.add_edge("ModelRoutingAgent", END)

    # Compile and run the graph
    graph_app = workflow.compile()

    # Optional: Plot the graph if the --plot flag is provided
    if plot:
        try:
            graph_image_path = "plot.png"
            # Create the Mermaid diagram as a PNG and save it
            graph_mermaid = graph_app.get_graph(xray=True).draw_mermaid_png()
            with open(graph_image_path, "wb") as f:
                f.write(graph_mermaid)
            img = Image.open(graph_image_path)
            img.show()
        except Exception as e:
            logger.warning(
                f"Graph visualization requires additional dependencies or there was an issue: {e}")

    # Main loop for user interaction
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            logger.info("Agent terminated by user.")
            break

        # Run the graph with the user query
        result = graph_app.invoke({"query": query})
        print("\n\n----------\n>>> Final Answer:\n\n", result["final_answer"])


if __name__ == "__main__":
    main()
