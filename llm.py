# llm.py
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_recommendation_explanation(
    user_mood: str,
    movies: list[dict]
) -> str:
    """
    This is the 'Generation' part of RAG.
    
    We've already RETRIEVED the relevant movies (via vector search).
    Now we GENERATE a human, personalized explanation of why they match.
    
    The LLM's job here is NOT to decide which movies to recommend —
    the vector database already did that. The LLM's job is to be
    a brilliant, articulate film critic who explains the connection
    between what the user wants to feel and what each film delivers.
    
    This is a crucial RAG design principle: use retrieval for WHAT,
    use generation for HOW TO EXPLAIN IT.
    """
    
    # Build a structured representation of the retrieved movies
    # to pass as context to the LLM
    movies_context = ""
    for i, movie in enumerate(movies, 1):
        movies_context += f"""
Movie {i}: {movie['title']} ({movie['year']})
Director: {movie['director']}
Genres: {movie['genres']}
Description: {movie['vibe_description']}
Match Score: {movie['similarity_score']}
---"""
    
    # ============================================================
    # PROMPT ENGINEERING — this is an art form
    # 
    # A good system prompt:
    # 1. Gives the LLM a clear persona and role
    # 2. Explains the task precisely
    # 3. Specifies the output format
    # 4. Provides constraints (what NOT to do)
    # 5. Gives the LLM the context it needs (the retrieved movies)
    # ============================================================
    
    system_prompt = """You are a deeply knowledgeable and passionate film critic with encyclopedic 
knowledge of cinema. Your gift is matching people to movies based on the emotional experience 
they're seeking — not just genre tags or plot summaries.

When given a user's mood and a set of pre-selected movies, your job is to write a warm, 
conversational, specific explanation of WHY each movie matches that mood. 

Rules:
- Write in second person ("you'll feel...", "this one gives you...")
- Be specific about scenes, moments, or qualities that match the mood — not generic praise
- Keep each movie explanation to 2-3 sentences
- If there's a reason the match is slightly imperfect, acknowledge it honestly
- Do NOT suggest movies outside the ones provided — the list was curated for them
- Format your response with the movie title as a header for each section"""

    user_message = f"""The user is looking for: "{user_mood}"

I've pre-selected these movies as potential matches based on semantic similarity:
{movies_context}

Please explain why each of these movies matches what the user is looking for. 
Write it like you're a film-savvy friend giving a personal recommendation."""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast and cheap, very capable
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,  
        # Temperature controls creativity vs consistency
        # 0.0 = deterministic, always same answer
        # 1.0 = creative, different each time
        # 0.7 is a good balance for recommendations
        max_tokens=800
    )
    
    return response.choices[0].message.content