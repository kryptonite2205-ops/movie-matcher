import os
import streamlit as st
from google import genai

def get_api_key():
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")

def generate_recommendation_explanation(user_mood: str, movies: list) -> str:
    client = genai.Client(api_key=get_api_key())

    movies_context = ""
    for i, movie in enumerate(movies, 1):
        movies_context += f"""
Movie {i}: {movie['title']} ({movie['year']})
Director: {movie['director']}
Genres: {movie['genres']}
Description: {movie['vibe_description']}
Match Score: {movie['similarity_score']}
---"""

    prompt = f"""You are a deeply knowledgeable and passionate film critic.
Your gift is matching people to movies based on the emotional experience they're seeking.

The user is looking for: "{user_mood}"

I've pre-selected these movies as potential matches:
{movies_context}

Write a warm, conversational explanation of WHY each movie matches what the user is looking for.
Rules:
- Write in second person ("you'll feel...", "this one gives you...")
- Be specific about qualities that match the mood
- Keep each movie explanation to 2-3 sentences
- Format with the movie title as a header for each section
- Do NOT suggest movies outside the ones provided"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"ERROR DETAILS: {str(e)}"
    return response.text