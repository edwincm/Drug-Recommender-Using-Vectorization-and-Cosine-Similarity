# Drug Recommender — TF-IDF & Cosine Similarity

An educational Flask application that explores text-similarity techniques on the **Drug Reviews (Drugs.com)** dataset from the UCI Machine Learning Repository.

> **Important:** This project is a machine-learning/NLP demonstration. It is **not** a clinical decision-support system and must not be used to select medication, diagnose a condition, or replace advice from a qualified healthcare professional.

## What the Project Does

Given a condition/symptom string, the application:

1. Loads the UCI Drug Reviews dataset.
2. Filters records whose `condition` contains the supplied query.
3. Builds TF-IDF vectors for the candidate review text and the user's query.
4. Calculates cosine similarity between each candidate review and the query vector.
5. Returns up to five unique drug names associated with the highest-ranked reviews.

The ranking is therefore based on **text similarity within the filtered dataset**, not medical efficacy or safety.

## Tech Stack

- Python
- Flask + Flask-CORS
- pandas / NumPy
- scikit-learn
- `ucimlrepo`
- TF-IDF vectorization
- Cosine similarity

## Run Locally

```bash
git clone https://github.com/edwincm/Drug-Recommender-Using-Vectorization-and-Cosine-Similarity.git
cd Drug-Recommender-Using-Vectorization-and-Cosine-Similarity
python -m venv .venv
```

Activate the virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask API:

```bash
python app.py
```

The API will be available at `http://127.0.0.1:5000` by default.

## API

### `POST /predict`

Request:

```json
{
  "symptom": "headache"
}
```

Successful response:

```json
{
  "recommendations": ["Drug A", "Drug B"]
}
```

Missing or blank `symptom` values return HTTP `400`.

## Tests

Install the development dependencies and run the focused regression tests:

```bash
pip install -r requirements-dev.txt
pytest -q
```

The automated tests cover query-aware ranking, the no-match case, and API input validation.

`benchmark_api.py` is a separate manual script for exercising a locally running API against example cases; it is not part of the unit-test suite.

## Example UI

<img width="740" alt="Drugs recommended for Headache" src="https://github.com/user-attachments/assets/74180eb5-ee83-4e04-8cfb-ec818c9df409">
<img width="740" alt="Drugs recommended for Vomiting" src="https://github.com/user-attachments/assets/bb5db3a4-1544-4f1e-943b-c96a42369834">

## Limitations

- The initial candidate set is based on literal condition-name matching.
- Ranking reflects lexical similarity in user reviews, not treatment suitability.
- The source dataset may contain noisy, incomplete, subjective, or outdated information.
- The application has not been clinically validated.
- Medication choice requires professional evaluation of diagnosis, contraindications, interactions, dosage, patient history, and other factors that this project does not model.
