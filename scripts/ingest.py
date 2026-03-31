# scripts/ingest.py
"""
Run this script ONCE to populate your ChromaDB with movie embeddings.
After this, the embeddings are saved to disk and persist between runs.

Usage: python scripts/ingest.py
"""
import sys
import os

# Add parent directory to path so we can import from embedder.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedder import ingest_movies

if __name__ == "__main__":
    print("Starting movie ingestion pipeline...")
    print("This will call the OpenAI Embeddings API for each movie.")
    print("Cost estimate: ~$0.001 total for 12 movies\n")
    
    collection = ingest_movies("data/movies.json")
    
    print(f"\nDatabase now contains {collection.count()} movies.")
    print("You can now run: streamlit run app.py")
    
    # Quick sanity check — search for something and verify it works
    print("\nRunning sanity check...")
    from embedder import search_movies
    results = search_movies("dark thriller with a twist ending", n_results=2)
    print(f"Test search 'dark thriller with a twist ending' returned:")
    for r in results:
        print(f"  - {r['title']} (similarity: {r['similarity_score']})")