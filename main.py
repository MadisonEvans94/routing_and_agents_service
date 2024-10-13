import json
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
        rag_response = self.qa_chain.invoke(query)
        logger.info(f"\n\n>>RAG response: {rag_response}\n\n")
        return rag_response



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
        response = self.llm_router.invoke(
            formatted_prompt)  # Get the AI response

        # Attempt to parse the response content as JSON
        try:
            route_decision = json.loads(response.content)
            datasource = route_decision.get('datasource')
            if datasource == "vectorstore":
                logger.info("RAG tool was invoked for the query: %s", query)
                # Now we return the retrieved context instead of a final response
                retrieved_context = self.rag_service.handle_query(query)
                return {'query': query, 'retrieved_context': retrieved_context}
            else:
                logger.info(
                    "Default LLM generation was used for the query: %s", query)
                # If not vectorstore, use LLM and return the generated result (no context)
                generated_result = self.llm(query)
                return {'query': query, 'retrieved_context': generated_result}
        except json.JSONDecodeError:
            logger.error("Failed to parse the routing decision as JSON.")
            return "Sorry, there was an error processing your request."


# 3. Final Model Agent to select between weak/strong models
class FinalModelAgent:
    def __init__(self):
        # Simulate weak and strong models for now
        self.weak_model = OpenAI(temperature=0.7)  # Example weak model
        self.strong_model = OpenAI(temperature=0)  # Example strong model

    def assess_query_complexity(self, query):
        """
        Simulate assessing the query complexity.
        For now, we'll assume any query longer than 10 words is 'complex' and short queries are 'simple'.
        """
        if len(query.split()) > 10:
            return "complex"
        else:
            return "simple"

    def route_model(self, query, context):
        """
        Route between weak and strong models using the user query and the retrieved context.
        """
        query_complexity = self.assess_query_complexity(query)

        # Combine the query and context into one final prompt
        final_prompt = f"Query: {query}\nContext: {context.get('result', 'No context')}"

        if query_complexity == "simple":
            logger.info("Using weak model for further processing.")
            # Pass combined query + context to the model
            return self.weak_model.invoke(final_prompt)
        else:
            logger.info("Using strong model for further processing.")
            # Pass combined query + context to the model
            return self.strong_model.invoke(final_prompt)


# Main function integrating RoutingAgent and FinalModelAgent
def main():
    # Ensure API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    logger.info("Starting the agent...")

    # Initialize routing agent and final model agent
    routing_agent = RoutingAgent(openai_api_key)
    final_model_agent = FinalModelAgent()  # Initialize the final model agent

    # Agent loop
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            logger.info("Agent terminated by user.")
            break

        # Route the query through the routing agent first
        routing_response = routing_agent.route_query(query)
        query = routing_response.get('query')
        retrieved_context = routing_response.get('retrieved_context')

        # Hand off the user query and retrieved context to the final model agent
        final_answer = final_model_agent.route_model(query, retrieved_context)
        print("Final Model Agent Answer:", final_answer)


if __name__ == "__main__":
    main()
