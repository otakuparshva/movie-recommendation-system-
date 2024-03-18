import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

st.set_page_config(
    page_title="CinePick",
    page_icon="🎬",
)

# Function to load movie data and initialize variables
def load_data():
    movies_df = pd.read_csv('movies.csv')
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))
    genres_matrix = vectorizer.fit_transform(movies_df['genres'])
    cosine_sim = cosine_similarity(genres_matrix, genres_matrix)
    return movies_df, vectorizer, genres_matrix, cosine_sim

# Function to recommend movies based on genres
def recommend_movies_by_genres(movies_df, vectorizer, genres_matrix, cosine_sim, exclude_genres=[], include_genres=[], n=1):
    # Vector representing include genres
    genres_vector = vectorizer.transform(include_genres)
    # Similarity scores
    sim_scores = cosine_similarity(genres_vector, genres_matrix).flatten()
    # Exclude specified genres (filter movie indices)
    exclude_indices = [i for i, genre in enumerate(movies_df['genres']) if any(exclude_genre in genre for exclude_genre in exclude_genres)]
    valid_indices = [i for i in range(len(sim_scores)) if i not in exclude_indices]
    if valid_indices:
        # Get movie indices based on similarity scores
        movie_indices = sorted(valid_indices, key=lambda x: sim_scores[x], reverse=True)[:n]
        return movies_df.iloc[movie_indices][['title', 'genres']]
    else:
        return movies_df.sample(n=n)[['title', 'genres']]

# Function to render dark mode
def render_dark_mode():
    st.markdown(
        """
        <style>
        .stApp {
            color: white;
            background-color: #1E1E1E;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to render light mode
def render_light_mode():
    st.markdown(
        """
        <style>
        .stApp {
            color: black;
            background-color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Main function to render the app
def main():
    # Load data
    movies_df, vectorizer, genres_matrix, cosine_sim = load_data()

    # Render dark mode or light mode based on user selection
    mode = st.sidebar.radio("Mode", ("Light", "Dark"))

    if mode == "Dark":
        render_dark_mode()
    else:
        render_light_mode()

    # Main app layout
    st.title('Movie Recommendation System')
    st.subheader('Customize Your Recommendations')
    st.write('Select genres to include or exclude in your movie recommendations.')
    
    # Genre selection
    include_genres_input = st.multiselect('Include Genres', options=sorted(vectorizer.vocabulary_.keys()), key='include_genres')
    exclude_genres_input = st.multiselect('Exclude Genres', options=sorted(vectorizer.vocabulary_.keys()), key='exclude_genres')

    # Number of recommendations selection
    num_recommendations = st.selectbox('How many movie recommendations do you want?', [1, 2, 3, 4, 5])

    # Get recommendation button
    if st.button('Get Recommendations'):
        recommendations = recommend_movies_by_genres(movies_df, vectorizer, genres_matrix, cosine_sim, include_genres=include_genres_input, exclude_genres=exclude_genres_input, n=num_recommendations)
        st.subheader('Top Recommendations')
        for index, row in recommendations.iterrows():
            st.write(f"**{row['title']}**")
            st.write(f"*Genres*: {row['genres']}\n")

# Run the app
if __name__ == "__main__":
    main()
