import pickle
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load pickled data
new = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Load CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        recommendations = get_movie_recommendations(movie_name)
        return render_template('index.html', movie_name=movie_name, recommendations=recommendations)
    return render_template('index.html', movie_name=None, recommendations=None)

def get_movie_recommendations(movie_name, num_recommendations=5):
    index = new[new['title'] == movie_name].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = [new.iloc[i[0]].title for i in distances[1:num_recommendations+1]]
    return recommended_movies

if __name__ == '__main__':
    app.run(debug=True)
