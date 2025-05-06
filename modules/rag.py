import gc

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import requests
from unstructured.partition.html import partition_html
from unstructured.partition.auto import partition
from loguru import logger
import re
import os
import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)



async def retrieve_rag(prompt, max_candidates=20, similarity_threshold=0.6):
    """Retrieve documents dynamically based on a similarity threshold."""
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    index = faiss.read_index("rag/rag_index.faiss")

    with open("rag/rag_index.pkl", "rb") as f:
        documents = pickle.load(f)

    query_embedding = model.encode([prompt])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

    # Fetch more candidates than you expect to use
    d, i = index.search(np.array(query_embedding), k=max_candidates)

    retrieved_docs = []
    for score, idx in zip(d[0], i[0]):
        if idx == -1:
            continue  # sometimes FAISS returns -1 if no match
        if score >= similarity_threshold:
            retrieved_docs.append(documents[idx])

    del model
    del index
    del documents
    gc.collect()

    return retrieved_docs


def create_rag_embeddings():
    documents = []

    url_documents = prepare_urls()
    xml_documents = prepare_documents()

    documents.extend(url_documents)
    documents.extend(xml_documents)

    save_document_index(documents)

    try:
        logger.info("Creating RAG embeddings")
        model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        vectors = model.encode(documents)

        # Normalize vectors to unit length
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine after normalization)
        index.add(vectors)
        logger.success("Embeddings created and normalized")
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        exit(1)

    save_embedding_index(index)

    logger.success("Rag DB successfully created")

def prepare_urls():
    try:
        documents = []
        logger.info("Loading URLs from rag/urls")
        with open("rag/urls", "r") as f:
            urls = [line.strip() for line in f if line.strip()]
            logger.info(urls)
            url_pattern = re.compile(r'^https?://')
            urls = [url for url in urls if url_pattern.match(url)]
            if not urls:
                raise ValueError("The 'rag/urls' file does not contain any valid URLs.")
            logger.success("URLs loaded")
        logger.info("Retrieving URLs")
        for url in urls:
            html = requests.get(url)
            elements = partition_html(text=html.text)

            document = " ".join(el.text for el in elements if el.text)
            documents.append(document)
            #for el in elements:
            #    documents.append(el.text)
        logger.success("URLs retrieved")
        return documents
    except Exception as e:
        logger.error(f"URL Retrieval failed: {e}")
        return []

def prepare_documents():
    try:
        documents = []
        logger.info("Loading documents from rag/docs")
        docs_folder = "rag/docs"
        for filename in os.listdir(docs_folder):
            filepath = os.path.join(docs_folder, filename)
            elements = partition(filename=filepath)
            document = " ".join(el.text for el in elements if el.text)
            documents.append(document)
            #for el in elements:
            #    documents.append(el.text)
        logger.success("Documents loaded")
        return documents
    except Exception as e:
        logger.error(f"Error loading Documents: {e}")
        return []

def save_document_index(documents):
    try:
        logger.info("Saving document index")
        with open("rag/rag_index.pkl", "wb") as f:
            pickle.dump(documents, f)
        logger.success("Document index saved")
    except Exception as e:
        logger.error(f"Error saving document index: {e}")
        exit(1)

def save_embedding_index(index):
    try:
        logger.info("Saving embedding index")
        faiss.write_index(index, "rag/rag_index.faiss")
        logger.success("Embedding index saved")
    except Exception as e:
        logger.error(f"Error saving embedding index: {e}")
        exit(1)


if __name__ == "__main__":
    create_rag_embeddings()