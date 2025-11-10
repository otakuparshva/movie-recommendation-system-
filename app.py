import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy import sparse
import heapq
import os
import random

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="MoviePick", page_icon="🎬", layout="centered")

random.seed(42)
np.random.seed(42)


def split_genres(x):
    if not isinstance(x, str) or not x:
        return []
    return [t.strip().lower() for t in x.split(',') if t.strip()]

@st.cache_data(show_spinner=False)
def load_data(path: str = 'movies.csv'):
    df = pd.read_csv(path)
    if 'genres' not in df.columns:
        df['genres'] = ''
    df['genres'] = df['genres'].fillna('').astype(str)
    vectorizer = CountVectorizer(tokenizer=split_genres, lowercase=False)
    matrix = vectorizer.fit_transform(df['genres'])
    return df.reset_index(drop=True), vectorizer, matrix

@st.cache_resource(show_spinner=False)
def build_autoencoder(input_dim: int, latent_dim: int = 32, epochs: int = 60, batch_size: int = 32):
    if not TF_AVAILABLE:
        return None
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(max(64, latent_dim * 4), activation='relu')(inp)
    x = layers.Dense(max(128, latent_dim * 6), activation='relu')(x)
    latent = layers.Dense(latent_dim, activation='relu')(x)
    x = layers.Dense(max(128, latent_dim * 6), activation='relu')(latent)
    x = layers.Dense(max(64, latent_dim * 4), activation='relu')(x)
    out = layers.Dense(input_dim, activation='sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)
    encoder = models.Model(inputs=inp, outputs=latent)
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy')
    return model, encoder

@st.cache_resource(show_spinner=False)
def train_autoencoder_if_needed(matrix, latent_dim=32, epochs=60):
    if not TF_AVAILABLE:
        return None
    X = matrix.toarray().astype('float32')
    model, encoder = build_autoencoder(X.shape[1], latent_dim=latent_dim, epochs=epochs)
    model.fit(X, X, epochs=epochs, batch_size=64, verbose=0)
    return encoder.predict(X, batch_size=64)


def top_n_indices(scores, n):
    if n <= 0:
        return []
    if len(scores) <= n:
        return list(range(len(scores)))
    return heapq.nlargest(n, range(len(scores)), key=lambda i: scores[i])


def recommend(df, vectorizer, matrix, include=None, exclude=None, n=5, use_nn_embeddings=True):
    include = [g.lower().strip() for g in (include or [])]
    exclude = [g.lower().strip() for g in (exclude or [])]
    all_genres = set(vectorizer.get_feature_names_out())
    include = [g for g in include if g in all_genres]
    exclude = [g for g in exclude if g in all_genres]
    N = df.shape[0]

    if include:
        include_vec = vectorizer.transform(include)
        include_scores = cosine_similarity(include_vec, matrix).mean(axis=0)
    else:
        include_scores = np.zeros(N)

    default_vec = vectorizer.transform(list(all_genres))
    default_scores = cosine_similarity(default_vec, matrix).mean(axis=0)

    final_scores = 0.85 * include_scores + 0.15 * default_scores

    if exclude:
        mask = df['genres'].apply(lambda g: any(e in g.lower() for e in exclude)).values
        final_scores = np.where(mask, -1.0, final_scores)

    if use_nn_embeddings and TF_AVAILABLE:
        try:
            embeddings = train_autoencoder_if_needed(matrix, latent_dim=min(64, matrix.shape[1] // 2 + 1), epochs=40)
            if embeddings is not None:
                if include:
                    include_vec_gen = vectorizer.transform(include).toarray().astype('float32')
                    encoder_input = np.vstack([matrix.toarray().astype('float32'), include_vec_gen])
                    encoded = embeddings
                    include_encoded = encoded[-len(include):] if len(include) > 0 else None
                    if include_encoded is not None:
                        sim = cosine_similarity(include_encoded, encoded).mean(axis=0)
                        final_scores = 0.6 * final_scores + 0.4 * sim
        except Exception:
            pass

    idxs = top_n_indices(final_scores, n * 6)
    if not idxs:
        return df.iloc[[]][['title', 'genres']]

    selected_pool = idxs
    k = min(n, len(selected_pool))
    chosen = np.random.choice(selected_pool, size=k, replace=False)
    return df.iloc[chosen][['title', 'genres']]


def apply_theme_css(theme: str):
    if theme == "Dark":
        css = """
        <style>
        div[data-testid="stAppViewContainer"] > div:first-child { background-color: #07070a; }
        section[data-testid="stSidebar"] { background-color: #0f1720 !important; color: #e6eef8; }
        div[data-testid="stAppViewContainer"] { color: #ecf0f3; }
        .main-title { text-align:center; font-size:44px; font-weight:800; margin-top:6px; color:#f7f9fb; }
        .subtitle { text-align:center; font-size:14px; color:#9aa4b2; margin-bottom:18px; }
        .block-container { padding-top:20px; max-width:960px; }
        .stButton>button { background: linear-gradient(90deg,#ff4b2b,#ff416c); color: white; border-radius:8px; font-weight:700; border: none; padding:8px 18px; }
        .stButton>button:hover { transform: translateY(-2px); }
        </style>
        """
    else:
        css = """
        <style>
        div[data-testid="stAppViewContainer"] > div:first-child { background-color: #ffffff; }
        div[data-testid="stAppViewContainer"] { color: #071023; }
        .main-title { text-align:center; font-size:44px; font-weight:800; margin-top:6px; }
        .subtitle { text-align:center; font-size:14px; color:#6b7280; margin-bottom:18px; }
        .block-container { padding-top:20px; max-width:960px; }
        .stButton>button { border-radius:8px; padding:8px 18px; font-weight:700; }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def main():
    df, vectorizer, matrix = load_data()
    st.sidebar.selectbox(" ", ["⚙️ Controls"]) 
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=1)
    apply_theme_css(theme)

    st.markdown("<div class='main-title'>🎬 MoviePick</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-Enhanced Movie Recommender — intern-level, production-friendly</div>", unsafe_allow_html=True)

    raw_genres = sorted(vectorizer.get_feature_names_out())
    display_genres = [g.title() for g in raw_genres]

    include_sel = st.multiselect('Include Genres (OR)', options=display_genres)
    exclude_sel = st.multiselect('Exclude Genres', options=display_genres)

    include_genres = [g.lower().strip() for g in include_sel]
    exclude_genres = [g.lower().strip() for g in exclude_sel]

    n = st.slider('Number of recommendations', min_value=1, max_value=12, value=5)
    use_nn = st.checkbox('Use neural embedding (autoencoder)', value=True and TF_AVAILABLE)

    if st.button("Get Recommendations 🎥"):
        with st.spinner('Crunching numbers...'):
            recs = recommend(df, vectorizer, matrix, include=include_genres, exclude=exclude_genres, n=n, use_nn_embeddings=use_nn)
        if recs.empty:
            st.warning("No matching recommendations found. Try other filters or reduce exclusions.")
        else:
            st.subheader("🎯 Top Picks For You")
            for _, row in recs.iterrows():
                st.markdown(f"**🎞️ {row['title']}**")
                st.markdown(f"<span style='color:gray'>Genres: {row['genres']}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
