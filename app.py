import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MoviePick", page_icon="🎬", layout="centered")

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv('movies.csv')
    df['genres'] = df['genres'].fillna('')
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','), lowercase=False)
    matrix = vectorizer.fit_transform(df['genres'])
    similarity = cosine_similarity(matrix)
    return df, vectorizer, matrix, similarity

def recommend(df, vectorizer, matrix, similarity, include=None, exclude=None, n=3):
    include = include or []
    exclude = exclude or []
    include_vec = vectorizer.transform(include) if include else np.zeros((1, matrix.shape[1]))
    include_score = cosine_similarity(include_vec, matrix).max(axis=0) if include else np.zeros(len(df))
    all_genres = set(vectorizer.get_feature_names_out())
    default_genres = list(all_genres - set(include) - set(exclude))
    default_vec = vectorizer.transform(default_genres) if default_genres else np.zeros((1, matrix.shape[1]))
    default_score = cosine_similarity(default_vec, matrix).mean(axis=0) if default_genres else np.zeros(len(df))
    exclude_idx = df.index[df['genres'].apply(lambda g: any(e in g for e in exclude))].tolist()
    final_score = 0.7 * include_score + 0.3 * default_score
    final_score[exclude_idx] = -1
    top_idx = np.argsort(final_score)[::-1][:n * 3]
    selected = np.random.choice(top_idx, size=min(n, len(top_idx)), replace=False)
    return df.iloc[selected][['title', 'genres']]

def main():
    df, vectorizer, matrix, similarity = load_data()
    st.markdown(
        """
        <style>
        .main-title {text-align:center;font-size:36px;font-weight:700;margin-top:10px;}
        .subtitle {text-align:center;font-size:18px;margin-bottom:20px;color:gray;}
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown("<div class='main-title'>🎬 MoviePick</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>AI-Powered Movie Recommendation with a Dash of Randomness</div>", unsafe_allow_html=True)

    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("<style>.stApp{background-color:#121212;color:#FFFFFF;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>.stApp{background-color:#FFFFFF;color:#000000;}</style>", unsafe_allow_html=True)

    include_genres = st.multiselect('Include Genres', sorted(vectorizer.get_feature_names_out()))
    exclude_genres = st.multiselect('Exclude Genres', sorted(vectorizer.get_feature_names_out()))

    if st.button("Get Recommendations 🎥"):
        recs = recommend(df, vectorizer, matrix, similarity, include=include_genres, exclude=exclude_genres, n=3)
        if recs.empty:
            st.warning("No matching recommendations found. Try adjusting your filters.")
        else:
            st.subheader("🎯 Top Picks For You")
            for _, row in recs.iterrows():
                st.markdown(f"**🎞️ {row['title']}**")
                st.markdown(f"<span style='color:gray'>Genres: {row['genres']}</span>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
