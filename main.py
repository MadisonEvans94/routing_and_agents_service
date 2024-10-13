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


def main():
    # Ensure API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    logger.info("Starting the agent...")

    # Load Spider-Man related documents
    urls = [
        "https://en.wikipedia.org/wiki/Spider-Man",
        "https://marvel.fandom.com/wiki/Peter_Parker_(Earth-616)",
    ]
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        doc = loader.load()
        docs.extend(doc)
    logger.info("Loaded Spider-Man documents.")

    # Split documents into chunks
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    split_docs = text_splitter.split_documents(docs)
    logger.info("Split documents into chunks.")

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    logger.info("Created embeddings for documents.")

    # Build vector store
    vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
    logger.info("Built vector store for retrieval.")

    # Set up retriever
    retriever = vectorstore.as_retriever()

    # Build RAG chain
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    logger.info("Initialized RetrievalQA chain.")

    # Define the routing class
    class RouteQuery(BaseModel):
        """Route a user query to the most relevant data source."""
        datasource: Literal["vectorstore", "web_search"] = Field(
            ..., description="Choose 'vectorstore' for Spider-Man topics, otherwise 'web_search'."
        )

    # Set up LLM for routing
    llm_router = ChatOpenAI(temperature=0)
    structured_llm_router = llm_router.with_structured_output(RouteQuery)

    system_message = """You are an expert at routing a user question to a vectorstore or web_search.
    The vectorstore contains documents related to Spider-Man.
    Use the vectorstore for questions on these topics. Otherwise, use web_search."""
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{question}"),
        ]
    )

    # Agent loop
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            logger.info("Agent terminated by user.")
            break

        # Decide whether to use vectorstore or web_search
        formatted_prompt = route_prompt.format(question=query)
        route_decision = structured_llm_router.invoke(
            formatted_prompt)  # Use invoke() instead of calling directly
        datasource = route_decision.datasource
        logger.info(f"Routing decision: {datasource}")

        if datasource == "vectorstore":
            # Use RAG
            answer = qa_chain.invoke(query)
            logger.info("Used RAG for answering.")
        else:
            # Use LLM directly
            answer = llm(query)
            logger.info("Used LLM directly for answering.")

        print("Answer:", answer)


if __name__ == "__main__":
    main()
