# agents/routing_agent.py

import logging
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from agents.rag_service import RAGService

logger = logging.getLogger(__name__)


class RAGAgent:
    def __init__(self, openai_api_key: str):
        # Initialize the LLM router
        self.llm_router = ChatOpenAI(temperature=0)
        self.system_message = """You are an expert at routing a user question to either a 'vectorstore' or 'web_search'.
        The vectorstore contains documents related to Spider-Man.
        Use the vectorstore for questions on these topics. Otherwise, use web_search.
        Your output should be a JSON object with the field 'datasource', where the value is either 'vectorstore' or 'web_search'."""
        self.route_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("human", "{question}")
        ])
        # Initialize the RAG service
        self.rag_service = RAGService(openai_api_key)
        self.llm = ChatOpenAI(temperature=0)

    def route_query(self, query: str) -> str:
        # Format and send the prompt for routing
        formatted_prompt = self.route_prompt.format(question=query)
        response = self.llm_router.invoke(formatted_prompt)

        try:
            route_decision = json.loads(response.content)
            datasource = route_decision.get('datasource')
            if datasource == "vectorstore":
                logger.info(f"RAG tool invoked for query: {query}")
                return self.rag_service.handle_query(query)
            else:
                logger.info(f"Default LLM used for query: {query}")
                return self.llm.invoke(query)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse routing decision: {str(e)}")
            return "Error in routing decision"
