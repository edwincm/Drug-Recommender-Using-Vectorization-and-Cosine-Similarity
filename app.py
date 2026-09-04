from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

app = Flask(__name__)

CORS(app)  # Enable CORS for all routes in the Flask app

# Load the dataset
from ucimlrepo import fetch_ucirepo
drug_reviews_drugs_com = fetch_ucirepo(id=462)
X = drug_reviews_drugs_com.data.features
y = drug_reviews_drugs_com.data.targets

# Convert features to dataframe
df = pd.DataFrame(X, columns=['drugName', 'condition', 'review'])

# Preprocess the data
df['review'] = df['review'].str.lower()

# Handle missing values in drugName
df['drugName'] = df['drugName'].fillna(df['drugName'].mode()[0])

# Function to recommend drugs for a given symptom
def recommend_drugs(symptom):
    # Drop rows with missing values required for filtering/ranking
    df_cleaned = df.dropna(subset=['condition', 'review', 'drugName'])

    # Filter dataframe for the given symptom/condition
    symptom_df = df_cleaned[
        df_cleaned['condition'].str.contains(symptom, case=False, regex=False)
    ]

    if symptom_df.empty:
        return ["Sorry, no drugs found for the given symptom."]

    # Fit the vectorizer on candidate reviews plus the user's query so the
    # reviews can be scored directly against what the user entered.
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    review_texts = symptom_df['review'].astype(str).tolist()
    tfidf_matrix = tfidf_vectorizer.fit_transform(review_texts + [symptom])

    review_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(review_vectors, query_vector).ravel()

    # Rank the most query-relevant reviews first.
    drug_indices = np.argsort(similarity_scores)[::-1]

    recommendations = []
    seen_drugs = set()
    for index in drug_indices:
        drug_name = symptom_df.iloc[index]['drugName']
        if drug_name not in seen_drugs:
            recommendations.append(drug_name)
            seen_drugs.add(drug_name)
        if len(recommendations) == 5:
            break

    return recommendations

@app.route('/', methods=['GET'])
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    symptom = str(data.get('symptom', '')).strip()

    if not symptom:
        return jsonify({'error': 'symptom is required'}), 400

    recommendations = recommend_drugs(symptom)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)