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
df['drugName'].fillna(df['drugName'].mode()[0], inplace=True)

# Function to recommend drugs for a given symptom
def recommend_drugs(symptom):
    # Drop rows with missing values in the 'condition' column
    df_cleaned = df.dropna(subset=['condition'])
    
    # Filter dataframe for the given symptom
    symptom_df = df_cleaned[df_cleaned['condition'].str.contains(symptom, case=False)]
    
    if symptom_df.empty:
        return ["Sorry, no drugs found for the given symptom."]
    
    # Compute TF-IDF vectors for reviews
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(symptom_df['review'])
    
    # Compute cosine similarity between symptom and drug reviewsß
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get indices of drugs with highest similarity to the symptom
    drug_indices = np.argsort(similarity[:, -1])[::-1]
    
    # Recommend top drugs
    recommendations = []
    seen_drugs = set()  # Set to store seen drugs
    for i in range(1, min(6, len(drug_indices))):
        index = drug_indices[i]
        drug_name = symptom_df.iloc[index]['drugName']
        if drug_name not in seen_drugs:
            recommendations.append(drug_name)
            seen_drugs.add(drug_name)
    
    return recommendations

@app.route('/', methods=['GET'])
def home():
    return "Flask server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptom = data['symptom']
    recommendations = recommend_drugs(symptom)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)