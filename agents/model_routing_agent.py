# agents/final_model_agent.py

import logging
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class ModelRoutingAgent:
    def __init__(self):
        # Initialize weak and strong models
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

    def assess_query_complexity(self, query: str) -> str:
        # Assess the complexity of the query
        if len(query.split()) > 10:
            return "complex"
        return "simple"

    def route_model(self, query: str, context: str) -> str:
        # Use the prompt template with the query and context
        final_prompt = self.prompt_template.format(
            query=query, context=context)

        # Route between weak and strong models based on query complexity
        query_complexity = self.assess_query_complexity(query)
        if query_complexity == "simple":
            logger.info(f"Using weak model for query: {query}")
            return self.weak_model.invoke(final_prompt)
        else:
            logger.info(f"Using strong model for query: {query}")
            return self.strong_model.invoke(final_prompt)
