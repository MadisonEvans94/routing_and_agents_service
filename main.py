import json
import os
import logging
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# 1. Separate RAG Service


class RAGService:
    def __init__(self, openai_api_key):
        # Load documents and set up vector store
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

        # Set up RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(temperature=0),
            retriever=self.retriever
        )
        logger.info("Initialized RAG service.")

    def handle_query(self, query):
        return self.qa_chain.invoke(query)

# 2. Routing Agent to decide between RAG or simple LLM


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

    def route_query(self, query):
        formatted_prompt = self.route_prompt.format(question=query)
        response = self.llm_router.invoke(formatted_prompt)  # Get the AI response

        # Attempt to parse the response content as JSON
        try:
            route_decision = json.loads(response.content)
            datasource = route_decision.get('datasource')
            if datasource == "vectorstore":
                logger.info("\n\n>>> RAG tool was invoked for the query: %s", query)
                return self.rag_service.handle_query(query)
            else:
                logger.info(
                    "\n\n>>> Default LLM generation was used for the query: %s", query)
                return self.llm(query)
        except json.JSONDecodeError:
            logger.error("Failed to parse the routing decision as JSON.")
            return "Sorry, there was an error processing your request."


# 3. Future placeholder: Final model router for weak/strong model logic


class FinalModelRouter:
    def __init__(self):
        # Placeholder for routing between weak and strong models
        self.weak_model = OpenAI(temperature=0.7)  # Example weak model
        self.strong_model = OpenAI(temperature=0)  # Example strong model

    def route_model(self, query_complexity):
        if query_complexity == "simple":
            return self.weak_model
        else:
            return self.strong_model


def main():
    # Ensure API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    logger.info("Starting the agent...")

    # Initialize routing agent
    routing_agent = RoutingAgent(openai_api_key)

    # Agent loop
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            logger.info("Agent terminated by user.")
            break

        # Route the query through the routing agent
        answer = routing_agent.route_query(query)
        print("Answer:", answer)


if __name__ == "__main__":
    main()
