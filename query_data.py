import logging
import sys

import chromadb
from langchain_community.embeddings import OllamaEmbeddings
from llama_index.core import Settings, PromptTemplate, StorageContext, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from populate_database import CHROMA_PATH

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://10.0.0.188:6008/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global query_engine
query_engine = None


def init_llm():
    llm = Ollama(model="llama3", request_timeout=3000.0)
    embed_model = OllamaEmbeddings(model="nomic-embed-text")

    Settings.llm = llm
    Settings.embed_model = embed_model


def init_index():
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_collection = chroma_client.get_collection("iollama")
    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection, embedding_function=Settings.embed_model
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create index from the existing vector store
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    return index


def init_query_engine(index):
    global query_engine

    # Custom prompt template
    template = (
        "You are a helpful assistant.\n\n"
        "Here is the context related to the query:\n"
        "-----------------------------------------\n"
        "{context_str}\n"
        "-----------------------------------------\n"
        "Considering the above information, please respond to the following inquiry\n\n"
        "Question: {query_str}\n\n"
    )
    qa_template = PromptTemplate(template)

    # Build query engine with custom template
    query_engine = index.as_query_engine(llm=Settings.llm, text_qa_template=qa_template, similarity_top_k=3)

    return query_engine


def chat(input_question):
    global query_engine

    response = query_engine.query(input_question)
    logging.info("got response from llm - %s", response)

    return response.response


if __name__ == "__main__":
    init_llm()
    INDEX = init_index()
    init_query_engine(INDEX)
    QUESTION = "what did it say about NDA?"
    REPLY = chat(QUESTION)
    print(REPLY)
