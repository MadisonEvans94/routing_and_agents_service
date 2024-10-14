import json
import os
import logging
from typing import Literal, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Define the state structure for the graph


class GraphState(Dict[str, Any]):
    query: str
    retrieved_context: str
    final_answer: str

# 1. Separate RAG Service


class RAGService:
    def __init__(self, openai_api_key):
        urls = [
            "https://en.wikipedia.org/wiki/Spider-Man",
            "https://marvel.fandom.com/wiki/Peter_Parker_(Earth-616)",
        ]
        self.docs = []
        for url in urls:
            loader = WebBaseLoader(url)
            doc = loader.load()
            self.docs.extend(doc)

        logger.info("Loaded Spider-Man documents.")

        # Split documents into chunks
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(self.docs)
        logger.info("Split documents into chunks.")

        # Create embeddings and build vector store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
        self.retriever = vectorstore.as_retriever()

        logger.info("Initialized RAG service.")

    def handle_query(self, query):
        # Retrieve raw text chunks without generation
        raw_docs = self.retriever.get_relevant_documents(query)
        raw_context = "\n\n".join([doc.page_content for doc in raw_docs])
        logger.info(f"\n\n>>Retrieved context: {raw_context}\n\n")
        return raw_context

# 2. Routing Agent node function


def routing_agent_node(state: GraphState):
    formatted_prompt = routing_agent.route_prompt.format(
        question=state["query"])
    response = routing_agent.llm_router.invoke(formatted_prompt)
    try:
        route_decision = json.loads(response.content)
        datasource = route_decision.get('datasource')
        if datasource == "vectorstore":
            logger.info(
                f"RAG tool was invoked for the query: {state['query']}")
            state["retrieved_context"] = routing_agent.rag_service.handle_query(
                state["query"])
        else:
            logger.info(
                f"Default LLM generation was used for the query: {state['query']}")
            state["retrieved_context"] = routing_agent.llm(state["query"])
    except json.JSONDecodeError:
        logger.error("Failed to parse the routing decision as JSON.")
        state["retrieved_context"] = "Error in routing decision"
    return state

# 3. Final Model Agent node function (Previously Missing)


def final_model_agent_node(state: GraphState):
    query = state["query"]
    context = state["retrieved_context"]

    query_complexity = final_model_agent.assess_query_complexity(query)

    # Combine the query and context into one final prompt using the PromptTemplate
    final_prompt = final_model_agent.prompt_template.format(
        query=query, context=context)

    # Route between weak and strong models based on query complexity
    if query_complexity == "simple":
        logger.info(f"Using weak model for query: {query}")
        state["final_answer"] = final_model_agent.weak_model.invoke(
            final_prompt)
    else:
        logger.info(f"Using strong model for query: {query}")
        state["final_answer"] = final_model_agent.strong_model.invoke(
            final_prompt)

    return state


# Initialize RoutingAgent and FinalModelAgent
routing_agent = None
final_model_agent = None

# Routing Agent class


class RoutingAgent:
    def __init__(self, openai_api_key):
        self.llm_router = ChatOpenAI(temperature=0)
        self.system_message = """You are an expert at routing a user question to either a 'vectorstore' or 'web_search'.
        The vectorstore contains documents related to Spider-Man.
        Use the vectorstore for questions on these topics. Otherwise, use web_search.
        Your output should be a JSON object with the field 'datasource', where the value is either 'vectorstore' or 'web_search'."""
        self.route_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("human", "{question}"),
        ])
        self.rag_service = RAGService(openai_api_key)
        self.llm = OpenAI(temperature=0)

# Final Model Agent class


class FinalModelAgent:
    def __init__(self):
        self.weak_model = OpenAI(temperature=0.7)
        self.strong_model = OpenAI(temperature=0)

        # Define a clear and concise prompt template for the final model
        self.prompt_template = PromptTemplate(
            input_variables=["context", "query"],
            template="""
            Context: {context}
            Question: {query}

            Please answer the question based only on the provided context in a concise manner.
            """
        )

    def assess_query_complexity(self, query):
        if len(query.split()) > 10:
            return "complex"
        return "simple"

# Main function


def main():
    # Ensure API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    logger.info("Starting the agent...")

    # Initialize RoutingAgent and FinalModelAgent
    global routing_agent, final_model_agent
    routing_agent = RoutingAgent(openai_api_key)
    final_model_agent = FinalModelAgent()

    # Initialize LangGraph StateGraph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("RoutingAgent", routing_agent_node)
    workflow.add_node("FinalModelAgent", final_model_agent_node)

    # Add edges between nodes
    workflow.add_edge(START, "RoutingAgent")
    workflow.add_edge("RoutingAgent", "FinalModelAgent")
    workflow.add_edge("FinalModelAgent", END)

    # Compile the graph
    graph_app = workflow.compile()

    # Agent loop
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            logger.info("Agent terminated by user.")
            break

        # Run the graph workflow with user query
        result = graph_app.invoke({"query": query})
        print("Final Model Agent Answer:", result["final_answer"])


if __name__ == "__main__":
    main()
