# app.py
import streamlit as st
from embedder import search_movies, ingest_movies
from llm import generate_recommendation_explanation

# ============================================================
# STREAMLIT FUNDAMENTALS
# 
# Streamlit re-runs the entire script top-to-bottom every time
# a user interacts with anything (types, clicks a button, etc.)
# 
# This means we use:
# - st.cache_resource: for things we want to load ONCE and keep
#   in memory (like database connections, loaded models)
# - st.session_state: for storing state between reruns
#   (like the user's query history or last results)
# ============================================================

st.set_page_config(
    page_title="Vibe Movie Matcher",
    page_icon="🎬",
    layout="centered"
)

# Initialize the database once, cached across all reruns and all users
@st.cache_resource
def initialize_database():
    """
    @st.cache_resource means this function runs once and the result
    is cached. Perfect for database connections and loaded models
    that are expensive to create but should be reused.
    """
    return ingest_movies("data/movies.json")

# Header
st.title("🎬 Vibe Movie Matcher")
st.markdown("""
*Forget genres and ratings. Describe the **feeling** you're after, 
and I'll find exactly the right film.*
""")

st.divider()

# Initialize DB silently in the background
with st.spinner("Loading movie database..."):
    initialize_database()

# ============================================================
# EXAMPLE MOODS — these serve two purposes:
# 1. Shows users what kinds of input work well
# 2. Demonstrates the semantic range of the system
# ============================================================

st.subheader("💭 What vibe are you in?")

example_moods = [
    "dark and gritty crime with a shocking twist",
    "makes me feel like the world is magical and beautiful",
    "psychological thriller that messes with your head",
    "deeply emotional but not sappy, makes you think about life",
    "intense and anxiety-inducing, can't look away",
    "weird and quirky but secretly profound"
]

# Let users click example moods as quick-start buttons
st.markdown("**Quick vibes:**")
cols = st.columns(2)
selected_example = None

for i, mood in enumerate(example_moods):
    col = cols[i % 2]
    if col.button(f"✨ {mood}", key=f"example_{i}", use_container_width=True):
        selected_example = mood

# Text input — pre-filled if user clicked an example
user_mood = st.text_area(
    label="Or describe your own vibe:",
    value=selected_example if selected_example else "",
    placeholder="e.g. 'dark psychological horror that stays with you for days'",
    height=100,
    help="Be descriptive! The more detail you give about the feeling you want, "
         "the better the matches will be."
)

# Number of results slider
n_results = st.slider(
    "How many recommendations?",
    min_value=1, max_value=5, value=3
)

# ============================================================
# THE MAIN ACTION
# ============================================================

if st.button("🔍 Find My Movies", type="primary", use_container_width=True):
    
    if not user_mood.strip():
        st.warning("Please describe a vibe first!")
        st.stop()
    
    # Two-phase process with visible progress feedback
    with st.spinner("🧠 Finding movies that match your vibe..."):
        # PHASE 1: Vector similarity search
        # Fast — just embedding one query and doing nearest-neighbor search
        matched_movies = search_movies(user_mood, n_results=n_results)
    
    with st.spinner("✍️ Crafting your personalized explanations..."):
        # PHASE 2: LLM generates explanation
        # Slower — waiting for API response
        explanation = generate_recommendation_explanation(user_mood, matched_movies)
    
    # ========================================================
    # DISPLAY RESULTS
    # ========================================================
    
    st.divider()
    st.subheader(f"🎯 Your Vibe Matches")
    st.markdown(f"*Based on: \"{user_mood}\"*")
    
    # Show the LLM's explanation (primary display)
    st.markdown(explanation)
    
    st.divider()
    
    # Show the raw similarity scores — educational and shows the system working
    with st.expander("🔬 See the raw similarity scores (how the matching worked)"):
        st.markdown("These scores show how semantically similar your mood description "
                   "was to each movie's vibe description, computed via vector similarity:")
        
        for movie in matched_movies:
            score = movie['similarity_score']
            # Color-code the score
            color = "green" if score > 0.5 else "orange" if score > 0.3 else "red"
            
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{movie['poster_emoji']} {movie['title']}** ({movie['year']})")
            col2.markdown(
                f"<span style='color:{color}; font-weight:bold'>{score:.1%} match</span>",
                unsafe_allow_html=True
            )
            
            # Visual progress bar for the score
            st.progress(min(score, 1.0))
    
    # Show full vibe descriptions that were used for matching
    with st.expander("📝 See the vibe descriptions used for matching"):
        st.markdown("These are the descriptions that were embedded and compared to your query:")
        for movie in matched_movies:
            st.markdown(f"**{movie['poster_emoji']} {movie['title']}**")
            st.markdown(f"*{movie['vibe_description']}*")
            st.markdown("---")

# Footer
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.8em'>"
    "Powered by OpenAI Embeddings + ChromaDB + GPT-4o-mini • RAG Architecture"
    "</div>",
    unsafe_allow_html=True
)