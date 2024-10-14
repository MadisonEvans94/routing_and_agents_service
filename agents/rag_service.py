# agents/rag_service.py

import logging
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, openai_api_key: str):
        # Load documents from URLs
        urls = [
            "https://en.wikipedia.org/wiki/Spider-Man",
            "https://marvel.fandom.com/wiki/Peter_Parker_(Earth-616)",
        ]
        self.docs = []
        for url in urls:
            loader = WebBaseLoader(url)
            doc = loader.load()
            self.docs.extend(doc)

        logger.info(">>> Loaded documents into vector store...")

        # Split documents into chunks
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(self.docs)
        logger.info(">>> Split documents into chunks...")

        # Create embeddings and build vector store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
        self.retriever = vectorstore.as_retriever()

        logger.info(">>> Initialized RAG service")

    def handle_query(self, query: str) -> str:
        # Retrieve raw text chunks without LLM generation
        # raw_docs = self.retriever.get_relevant_documents(query)
        # raw_context = "\n\n".join([doc.page_content for doc in raw_docs])
        # logger.info(f"Retrieved context: {raw_context}")
        # return raw_context
        return self.retriever.invoke(query)
