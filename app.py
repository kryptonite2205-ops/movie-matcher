import streamlit as st
from embedder import search_movies, ingest_movies
from llm import generate_recommendation_explanation

st.set_page_config(
    page_title="Vibe Movie Matcher",
    page_icon="🎬",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Crimson+Text:ital,wght@0,400;0,600;1,400&display=swap');

/* Background: dark cinematic collage feel */
.stApp {
    background:
        radial-gradient(ellipse at 10% 20%, rgba(139,0,0,0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(0,0,139,0.15) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(75,0,130,0.1) 0%, transparent 70%),
        linear-gradient(135deg, #0a0a0f 0%, #111118 40%, #0d0d15 100%);
    background-attachment: fixed;
}

/* Film strip pattern overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 60px,
            rgba(255,255,255,0.012) 60px,
            rgba(255,255,255,0.012) 61px
        ),
        repeating-linear-gradient(
            90deg,
            transparent,
            transparent 80px,
            rgba(255,255,255,0.008) 80px,
            rgba(255,255,255,0.008) 81px
        );
    pointer-events: none;
    z-index: 0;
}

/* Global font */
html, body, [class*="css"], .stMarkdown, p, div {
    font-family: 'Crimson Text', 'Bell MT', Georgia, serif !important;
}

h1, h2, h3, .big-title {
    font-family: 'Playfair Display', 'Bell MT', Georgia, serif !important;
}

/* Main title */
.main-title {
    font-family: 'Playfair Display', 'Bell MT', Georgia, serif !important;
    font-size: 3.2rem !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, #f5c518, #ff6b35, #c850c0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: 2px;
    margin-bottom: 0.2rem;
    text-shadow: none;
}

.subtitle {
    font-family: 'Crimson Text', 'Bell MT', Georgia, serif !important;
    font-size: 1.15rem !important;
    color: #a0a0b0 !important;
    text-align: center;
    font-style: italic;
    margin-bottom: 2rem;
    letter-spacing: 0.5px;
}

/* Movie poster card grid */
.poster-strip {
    display: flex;
    gap: 8px;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1rem 0 2rem 0;
    opacity: 0.6;
}

.poster-card {
    width: 52px;
    height: 78px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    border: 1px solid rgba(255,255,255,0.1);
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(4px);
}

/* Quick vibe buttons */
.stButton > button {
    background: rgba(245, 197, 24, 0.08) !important;
    color: #f5c518 !important;
    border: 1px solid rgba(245, 197, 24, 0.3) !important;
    border-radius: 30px !important;
    font-family: 'Crimson Text', 'Bell MT', Georgia, serif !important;
    font-size: 0.95rem !important;
    font-style: italic;
    padding: 0.5rem 1rem !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(8px) !important;
}

.stButton > button:hover {
    background: rgba(245, 197, 24, 0.2) !important;
    border-color: #f5c518 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 20px rgba(245, 197, 24, 0.2) !important;
}

/* Selected vibe button */
.stButton > button:focus {
    background: rgba(245, 197, 24, 0.25) !important;
    border-color: #f5c518 !important;
    box-shadow: 0 0 0 2px rgba(245, 197, 24, 0.4) !important;
}

/* Find Movies button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #f5c518, #ff6b35) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 30px !important;
    font-family: 'Playfair Display', 'Bell MT', Georgia, serif !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    font-style: normal !important;
    letter-spacing: 1px !important;
    padding: 0.7rem 2rem !important;
    box-shadow: 0 4px 25px rgba(245, 197, 24, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 35px rgba(245, 197, 24, 0.4) !important;
}

/* Text area */
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(245, 197, 24, 0.2) !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    font-family: 'Crimson Text', 'Bell MT', Georgia, serif !important;
    font-size: 1rem !important;
    font-style: italic !important;
}

.stTextArea > div > div > textarea:focus {
    border-color: rgba(245, 197, 24, 0.6) !important;
    box-shadow: 0 0 20px rgba(245, 197, 24, 0.1) !important;
}

/* Slider */
.stSlider > div > div > div > div {
    background: linear-gradient(135deg, #f5c518, #ff6b35) !important;
}

/* Results section */
.result-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(245, 197, 24, 0.15);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}

/* Divider */
hr {
    border-color: rgba(245, 197, 24, 0.15) !important;
    margin: 1.5rem 0 !important;
}

/* Section headers */
.section-label {
    font-family: 'Playfair Display', 'Bell MT', Georgia, serif !important;
    font-size: 1.1rem !important;
    color: #f5c518 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
    margin-bottom: 0.8rem !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'Crimson Text', 'Bell MT', Georgia, serif !important;
    color: #a0a0b0 !important;
    font-style: italic !important;
}

/* Warning/success messages */
.stAlert {
    border-radius: 12px !important;
    font-family: 'Crimson Text', 'Bell MT', Georgia, serif !important;
}

/* Hide debug key message after everything works */
.debug-hide {
    display: none;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: rgba(245,197,24,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Initialize DB ─────────────────────────────────────────────────────────────
@st.cache_resource
def initialize_database():
    return ingest_movies("data/movies.json")

with st.spinner("🎞️ Loading the cinematheque..."):
    initialize_database()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Vibe Movie Matcher</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Forget genres and ratings — describe the <em>feeling</em> you\'re after</div>',
    unsafe_allow_html=True
)

# Movie poster strip decoration
st.markdown("""
<div class="poster-strip">
  <div class="poster-card">🔪</div>
  <div class="poster-card">🃏</div>
  <div class="poster-card">🪜</div>
  <div class="poster-card">💭</div>
  <div class="poster-card">🌆</div>
  <div class="poster-card">🥁</div>
  <div class="poster-card">🥯</div>
  <div class="poster-card">👁️</div>
  <div class="poster-card">🏨</div>
  <div class="poster-card">🔨</div>
  <div class="poster-card">🌸</div>
  <div class="poster-card">💈</div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Session state for selected vibe ──────────────────────────────────────────
# THIS IS THE BUG FIX:
# When user clicks a quick vibe button, we store it in session_state.
# session_state persists across reruns, so the value is available
# when the Find My Movies button is clicked.
if "selected_vibe" not in st.session_state:
    st.session_state.selected_vibe = ""

# ── Quick Vibe Buttons ────────────────────────────────────────────────────────
st.markdown('<div class="section-label">✦ Quick Vibes</div>', unsafe_allow_html=True)

example_moods = [
    "dark and gritty crime with a shocking twist",
    "makes me feel like the world is magical and beautiful",
    "psychological thriller that messes with your head",
    "deeply emotional but not sappy, makes you think about life",
    "intense and anxiety-inducing, can't look away",
    "weird and quirky but secretly profound"
]

cols = st.columns(2)
for i, mood in enumerate(example_moods):
    col = cols[i % 2]
    if col.button(f"✨ {mood}", key=f"vibe_{i}", use_container_width=True):
        # BUG FIX: store clicked vibe in session_state
        st.session_state.selected_vibe = mood

# ── Text Input ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label" style="margin-top:1.2rem">✦ Or Describe Your Own</div>',
            unsafe_allow_html=True)

user_typed = st.text_area(
    label="",
    value=st.session_state.selected_vibe,  # Pre-fill with clicked vibe
    placeholder="e.g. 'dark psychological horror that stays with you for days...'",
    height=100,
    key="mood_input"
)

# Update session state if user types manually
if user_typed != st.session_state.selected_vibe:
    st.session_state.selected_vibe = user_typed

# Show which vibe is currently selected
if st.session_state.selected_vibe:
    st.markdown(
        f'<div style="color:#f5c518; font-style:italic; font-size:0.9rem; margin-top:-0.5rem;">'
        f'🎯 Current vibe: "{st.session_state.selected_vibe[:60]}{"..." if len(st.session_state.selected_vibe) > 60 else ""}"'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("")

n_results = st.slider("How many recommendations?", min_value=1, max_value=5, value=3)

st.markdown("")

# ── Find Button ───────────────────────────────────────────────────────────────
find_clicked = st.button("🔍 Find My Movies", type="primary", use_container_width=True)

if find_clicked:
    # BUG FIX: use session_state value, not just text area value
    final_mood = st.session_state.selected_vibe.strip()

    if not final_mood:
        st.warning("✦ Please describe a vibe or click a quick suggestion first!")
        st.stop()

    with st.spinner("🧠 Scanning the cinematheque for your vibe..."):
        matched_movies = search_movies(final_mood, n_results=n_results)

    with st.spinner("✍️ Your personal film critic is writing your recommendations..."):
        explanation = generate_recommendation_explanation(final_mood, matched_movies)

    # ── Results ───────────────────────────────────────────────────────────────
    st.divider()

    st.markdown(
        f'<div style="font-family: Playfair Display, Bell MT, Georgia, serif; '
        f'font-size:1.6rem; color:#f5c518; font-weight:700; margin-bottom:0.3rem;">'
        f'✦ Your Matches</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="color:#888; font-style:italic; margin-bottom:1rem;">'
        f'Based on: "{final_mood}"</div>',
        unsafe_allow_html=True
    )

    # Main explanation from LLM
    st.markdown(
        f'<div class="result-box">{explanation}</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # Similarity scores
    with st.expander("🔬 See similarity scores"):
        st.markdown("*How closely your mood matched each film's vibe description:*")
        for movie in matched_movies:
            score = movie['similarity_score']
            color = "#4CAF50" if score > 0.5 else "#FF9800" if score > 0.3 else "#f44336"
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**{movie['poster_emoji']} {movie['title']}** ({movie['year']})")
            col2.markdown(
                f"<span style='color:{color}; font-weight:bold'>{score:.1%}</span>",
                unsafe_allow_html=True
            )
            st.progress(min(score, 1.0))

    with st.expander("📝 See vibe descriptions used for matching"):
        for movie in matched_movies:
            st.markdown(f"**{movie['poster_emoji']} {movie['title']}**")
            st.markdown(f"*{movie['vibe_description']}*")
            st.markdown("---")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#444; font-size:0.8rem; font-style:italic;'>"
    "Powered by Gemini Embeddings · ChromaDB · RAG Architecture"
    "</div>",
    unsafe_allow_html=True
)