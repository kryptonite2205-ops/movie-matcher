import os
import json
import chromadb
import streamlit as st
from openai import OpenAI

def get_api_key():
    try:
        key = st.secrets["OPENAI_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=get_api_key())
chroma_client = chromadb.EphemeralClient()


def get_or_create_collection():
    return chroma_client.get_or_create_collection(
        name="movies",
        metadata={"hnsw:space": "cosine"}
    )


def embed_text(text: str) -> list:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def ingest_movies(movies_path: str = "data/movies.json"):
    collection = get_or_create_collection()

    if collection.count() > 0:
        print(f"Collection already has {collection.count()} movies. Skipping ingestion.")
        return collection

    with open(movies_path, "r") as f:
        movies = json.load(f)

    print(f"Ingesting {len(movies)} movies into ChromaDB...")

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for movie in movies:
        print(f"  Embedding: {movie['title']}...")
        embedding = embed_text(movie["vibe_description"])

        ids.append(movie["id"])
        embeddings.append(embedding)
        documents.append(movie["vibe_description"])
        metadatas.append({
            "title": movie["title"],
            "year": str(movie["year"]),
            "genres": ", ".join(movie["genres"]),
            "director": movie["director"],
            "poster_emoji": movie["poster_emoji"]
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print(f"Successfully ingested {len(movies)} movies!")
    return collection


def search_movies(query: str, n_results: int = 3) -> list:
    collection = get_or_create_collection()
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )

    movies = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance

        movies.append({
            "title": results["metadatas"][0][i]["title"],
            "year": results["metadatas"][0][i]["year"],
            "genres": results["metadatas"][0][i]["genres"],
            "director": results["metadatas"][0][i]["director"],
            "poster_emoji": results["metadatas"][0][i]["poster_emoji"],
            "vibe_description": results["documents"][0][i],
            "similarity_score": round(similarity, 3)
        })

    return movies