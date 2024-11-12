import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(
    page_title="MoviePick",
    page_icon="🎬",
)

def load():
    df = pd.read_csv('movies.csv')
    vect = CountVectorizer(tokenizer=lambda x: x.split(','))
    matrix = vect.fit_transform(df['genres'])
    sim = cosine_similarity(matrix, matrix)
    return df, vect, matrix, sim

def get_recs(df, vect, matrix, sim, inc=[], exc=[], n=3):
    if inc:
        inc_vec = vect.transform(inc)
        inc_scores = cosine_similarity(inc_vec, matrix).max(axis=0)
    else:
        inc_scores = np.zeros(len(df))

    all_genres = vect.get_feature_names_out()
    def_genres = list(set(all_genres) - set(inc) - set(exc))
    if def_genres:
        def_vec = vect.transform(def_genres)
        def_scores = cosine_similarity(def_vec, matrix).mean(axis=0)
    else:
        def_scores = np.zeros(len(df))

    exc_idx = [i for i, genre in enumerate(df['genres']) if any(ex in genre for ex in exc)]
    scores = 0.7 * inc_scores + 0.3 * def_scores
    scores = [score if idx not in exc_idx else -1 for idx, score in enumerate(scores)]

    top_idx = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)[:n*2]
    random_indices = np.random.choice(top_idx, n, replace=False)
    return df.iloc[random_indices][['title', 'genres']]

def main():
    df, vect, matrix, sim = load()

    mode = st.sidebar.radio("Mode", ("Light", "Dark"))
    if mode == "Dark":
        st.markdown("<style>.stApp {color: white; background-color: #1E1E1E;}</style>", unsafe_allow_html=True)
    else:
        st.markdown("<style>.stApp {color: black; background-color: white;}</style>", unsafe_allow_html=True)

    st.title("MoviePick - Randomized Movie Recommendation")
    st.subheader("Choose genres to customize your recommendations with added randomness.")

    inc_input = st.multiselect('Include Genres', options=sorted(vect.get_feature_names_out()), key='inc_genres')
    exc_input = st.multiselect('Exclude Genres', options=sorted(vect.get_feature_names_out()), key='exc_genres')

    if st.button("Get Recommendations"):
        recs = get_recs(df, vect, matrix, sim, inc=inc_input, exc=exc_input, n=3)

        st.subheader("Top Recommendations")
        for index, row in recs.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"*Genres*: {row['genres']}")

if __name__ == "__main__":
    main()
