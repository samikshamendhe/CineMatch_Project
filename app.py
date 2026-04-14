import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# 🎨 CLEAN MINIMAL UI
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
}

/* Title */
h1 {
    color: #222 !important;
    text-align: center;
    font-weight: 700;
}

/* Subtitle */
p {
    color: #444 !important;
    text-align: center;
}

/* Input */
.stNumberInput input {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
}

.stButton > button {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    color: black !important;
    font-size: 18px;
    font-weight: bold;
    padding: 12px 25px;
    border-radius: 12px;
    border: none;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.2);
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #43e97b, #38f9d7);
    color: black;
    transform: scale(1.05);
}

/* Container */
.block-container {
    background: rgba(255,255,255,0.8);
    padding: 2rem;
    border-radius: 15px;
}

/* Text */
h2, h3 {
    color: #333 !important;
}
</style>
""", unsafe_allow_html=True)

# 🎬 TITLE
st.title("🎬 CineMatch")
st.markdown(
    "<h2 style='text-align: center; font-size: 20px; color: #333;'>AI Movie Recommendation System using Collaborative Filtering</h2>",
    unsafe_allow_html=True
)

# 📂 LOAD DATA
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# 🎯 PREPARE DATA
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# 🔀 TRAIN MODEL
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

# 🔍 RECOMMEND FUNCTION
def recommend_movies(user_id, n=5):
    movie_ids = movies['movieId'].unique()
    predictions = []

    for movie_id in movie_ids:
        pred = model.predict(user_id, movie_id)
        predictions.append((movie_id, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:n]

    recommended_titles = []
    for movie_id, _ in top_movies:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        recommended_titles.append(title)

    return recommended_titles


# 🔥 ADD THIS BELOW IT
def cold_start_recommendation(n=5):
    popular_movies = ratings.groupby('movieId')['rating'].count().sort_values(ascending=False)
    top_movie_ids = popular_movies.head(n).index
    recommended = movies[movies['movieId'].isin(top_movie_ids)]['title']
    return recommended.tolist()

# 🎯 INPUT
user_id = st.number_input("Enter User ID", min_value=1, step=1)

# 🔘 BUTTON
if st.button("Get Recommendations"):

    # 🔴 CASE 1: New User (Cold Start)
    if user_id not in ratings['userId'].values:

        st.subheader("🆕 New User Detected ")
        st.write("Showing popular movies:")

        recommendations = cold_start_recommendation(5)

        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")

    # 🟢 CASE 2: Existing User
    else:

        # 🎬 Show liked movies
        st.subheader("🎬 Movies You Liked:")

        user_data = ratings[ratings['userId'] == user_id]
        merged = pd.merge(user_data, movies, on='movieId')

        liked_movies = merged[merged['rating'] >= 4]

        for movie in liked_movies['title'].head(5):
            st.write(movie)

        # 🎯 Show recommendations
        recommendations = recommend_movies(user_id)

        st.subheader("🎯 Top Recommended Movies:")

        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")