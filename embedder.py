# embedder.py
import os
import json
import chromadb
from openai import OpenAI

import streamlit as st
def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=get_api_key())

# ============================================================
# WHAT IS CHROMADB?
# ChromaDB is a vector database. Unlike a regular database that
# stores rows of data and lets you query by exact match,
# ChromaDB stores vectors (lists of floats) and lets you query
# by SIMILARITY — "give me the 3 vectors most similar to this one."
# 
# It runs entirely in-memory or on disk locally.
# No Docker, no server, no cloud account needed for development.
# ============================================================



# PersistentClient saves the database to disk so you don't
# have to re-embed every time you restart the app
chroma_client = chromadb.EphemeralClient()

def get_or_create_collection():
    """
    A ChromaDB 'collection' is like a table in a regular database,
    except instead of rows indexed by ID, it stores vectors indexed
    by similarity. Each item has: an ID, a vector, the original text,
    and optional metadata.
    
    get_or_create_collection means: if it already exists (from a
    previous run), use it. Otherwise create a fresh one.
    """
    return chroma_client.get_or_create_collection(
        name="movies",
        # cosine similarity is better than euclidean for text embeddings
        # because it measures angle (meaning direction) not raw distance
        metadata={"hnsw:space": "cosine"}
    )

def embed_text(text: str) -> list[float]:
    """
    Convert any text string into a vector of 1536 floats using
    OpenAI's text-embedding-3-small model.
    
    This model was trained to place semantically similar texts
    close together in this 1536-dimensional space. The numbers
    themselves are meaningless in isolation — what matters is
    the RELATIONSHIP (distance/angle) between vectors.
    
    Why text-embedding-3-small?
    - Cheap: ~$0.00002 per 1000 tokens
    - Fast: single API call, returns in ~200ms
    - Excellent quality for semantic search tasks
    """
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding  # Returns list of 1536 floats

def ingest_movies(movies_path: str = "data/movies.json"):
    """
    One-time setup: load all movies, embed their descriptions,
    and store in ChromaDB.
    
    This is the 'offline' phase of RAG. You do this once.
    At query time, you just compare the user's query vector
    against these pre-computed vectors — no re-embedding needed.
    """
    collection = get_or_create_collection()
    
    # Don't re-ingest if we already have data
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
        
        # This is the text we embed — the rich vibe description
        # NOT just the title. The title "Se7en" contains no semantic
        # information about mood. The vibe description does.
        embedding = embed_text(movie["vibe_description"])
        
        ids.append(movie["id"])
        embeddings.append(embedding)
        documents.append(movie["vibe_description"])
        
        # Metadata is stored alongside vectors and returned with results
        # It's not used for similarity search, but for displaying results
        metadatas.append({
            "title": movie["title"],
            "year": str(movie["year"]),
            "genres": ", ".join(movie["genres"]),
            "director": movie["director"],
            "poster_emoji": movie["poster_emoji"]
        })
    
    # Batch insert — much more efficient than inserting one at a time
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✅ Successfully ingested {len(movies)} movies!")
    return collection

def search_movies(query: str, n_results: int = 3) -> list[dict]:
    """
    The core retrieval step of RAG.
    
    1. Convert the user's query into a vector (same embedding model!)
    2. Ask ChromaDB: "which stored vectors are closest to this?"
    3. Return those movies + their similarity scores
    
    CRITICAL: You MUST use the same embedding model for queries
    as you used for ingestion. Different models produce vectors in
    different spaces — comparing them would be meaningless.
    """
    collection = get_or_create_collection()
    
    # Embed the user's mood query — same model as ingestion
    query_embedding = embed_text(query)
    
    # ChromaDB computes cosine similarity between query_embedding
    # and every stored embedding, returns the n_results most similar
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"]
    )
    
    # Restructure results into clean list of dicts
    movies = []
    for i in range(len(results["ids"][0])):
        # Distance in cosine space: 0 = identical, 2 = opposite
        # Convert to similarity score: 1 = perfect match, 0 = no match
        distance = results["distances"][0][i]
        similarity = 1 - distance  # Higher is better
        
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