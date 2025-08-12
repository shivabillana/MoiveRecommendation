from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Initialize the Flask app
app = Flask(__name__)

# --- ONE-TIME SETUP: Load and process the data ---
# This part runs only once when the server starts.
print("Loading dataset and training model... This may take a moment.")

# Load the dataset
try:
    movies_data = pd.read_csv('movies.csv')
except FileNotFoundError:
    print("Error: 'movies.csv' not found.")
    exit()

# Preprocess and combine features
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

movies_data['combined_features'] = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Vectorize the text
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(movies_data['combined_features'])

# Calculate cosine similarity
similarity = cosine_similarity(feature_vectors)

list_of_all_titles = movies_data['title'].tolist()

print("Model trained and ready.")
# --- END OF ONE-TIME SETUP ---


# --- Define the routes (web pages) ---

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handles the form submission and displays recommendations."""
    # Get the movie name from the form submission
    movie_name = request.form.get('movie_name')
    
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if not find_close_match:
        return render_template('recommendations.html', movie_name=movie_name, movies=None)
    
    close_match = find_close_match[0]
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    similarity_score = list(enumerate(similarity[index_of_the_movie]))
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    
    # Get the top 20 recommendations
    recommended_movies = []
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        if index != index_of_the_movie:
            recommended_movies.append(title_from_index)
            i += 1
        if i > 20:
            break
            
    return render_template('recommendations.html', movie_name=close_match, movies=recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)