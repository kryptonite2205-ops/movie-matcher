import os
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


def generate_recommendation_explanation(user_mood: str, movies: list) -> str:

    movies_context = ""
    for i, movie in enumerate(movies, 1):
        movies_context += f"""
Movie {i}: {movie['title']} ({movie['year']})
Director: {movie['director']}
Genres: {movie['genres']}
Description: {movie['vibe_description']}
Match Score: {movie['similarity_score']}
---"""

    system_prompt = """You are a deeply knowledgeable and passionate film critic with encyclopedic 
knowledge of cinema. Your gift is matching people to movies based on the emotional experience 
they're seeking, not just genre tags or plot summaries.

When given a user's mood and a set of pre-selected movies, your job is to write a warm, 
conversational, specific explanation of WHY each movie matches that mood. 

Rules:
- Write in second person ("you'll feel...", "this one gives you...")
- Be specific about scenes, moments, or qualities that match the mood
- Keep each movie explanation to 2-3 sentences
- If there's a reason the match is slightly imperfect, acknowledge it honestly
- Do NOT suggest movies outside the ones provided
- Format your response with the movie title as a header for each section"""

    user_message = f"""The user is looking for: "{user_mood}"

I've pre-selected these movies as potential matches based on semantic similarity:
{movies_context}

Please explain why each of these movies matches what the user is looking for.
Write it like you're a film-savvy friend giving a personal recommendation."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=800
    )

    return response.choices[0].message.content