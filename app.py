import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

st.set_page_config(
    page_title="CinePick",
    page_icon="🍿",
)

movies_df = pd.read_csv('movies.csv')

vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
genres_matrix = vectorizer.fit_transform(movies_df['genres'])

cosine_sim = cosine_similarity(genres_matrix, genres_matrix)

def recommend_movies_by_genres(exclude_genres=[], include_genres=[], cosine_sim=cosine_sim, n=10):
    # vector representing include genres
    genres_vector = vectorizer.transform(include_genres)

    # similarity scores
    sim_scores = cosine_similarity(genres_vector, genres_matrix).flatten()

    # exclude specified genres
    for exclude_genre in exclude_genres:
        exclude_indices = [i for i, genre in enumerate(movies_df['genres']) if exclude_genre in genre]
        sim_scores[exclude_indices] = -1  # -1 for excluded genres

    # filter out indices which are out of bounds
    valid_indices = [i for i in range(len(sim_scores)) if i < len(movies_df)]

    # if no valid indices, recommend a random movie
    if not valid_indices:
        random_movie = movies_df.sample()
        return random_movie

    # get movie indices based on similarity scores
    movie_indices = sorted(valid_indices, key=lambda x: sim_scores[x], reverse=True)[:n]

    return movies_df.iloc[movie_indices][['title', 'genres']]

def main():
    st.title('Movie Recommendation System')

    st.subheader('Customize Your Recommendations')
    st.write('Select genres to include or exclude in your movie recommendations.')

    include_genres_input = st.multiselect('Include Genres', options=sorted(vectorizer.vocabulary_.keys()), key='include_genres')
    exclude_genres_input = st.multiselect('Exclude Genres', options=sorted(vectorizer.vocabulary_.keys()), key='exclude_genres')

    if st.button('Get Recommendations'):
        recommendations = recommend_movies_by_genres(include_genres=include_genres_input, exclude_genres=exclude_genres_input)
        st.subheader('Top Recommendations')
        for index, row in recommendations.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"*Genres*: {row['genres']}\n")

if __name__ == "__main__":
    main()
