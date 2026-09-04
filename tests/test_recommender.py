import importlib
from pathlib import Path
import sys
import types

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DummyFlask:
    def __init__(self, *args, **kwargs):
        pass

    def route(self, *args, **kwargs):
        return lambda func: func

    def run(self, *args, **kwargs):
        pass


def load_app_with_stubbed_dependencies():
    flask_module = types.ModuleType("flask")
    flask_module.Flask = DummyFlask
    flask_module.request = types.SimpleNamespace(get_json=lambda **kwargs: {})
    flask_module.jsonify = lambda value: value

    flask_cors_module = types.ModuleType("flask_cors")
    flask_cors_module.CORS = lambda app: app

    sample_features = pd.DataFrame(
        [
            {"drugName": "Sample", "condition": "headache", "review": "sample review"},
        ]
    )
    sample_targets = pd.DataFrame()
    ucimlrepo_module = types.ModuleType("ucimlrepo")
    ucimlrepo_module.fetch_ucirepo = lambda id: types.SimpleNamespace(
        data=types.SimpleNamespace(features=sample_features, targets=sample_targets)
    )

    sys.modules["flask"] = flask_module
    sys.modules["flask_cors"] = flask_cors_module
    sys.modules["ucimlrepo"] = ucimlrepo_module
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_recommendations_rank_reviews_against_the_user_query():
    app = load_app_with_stubbed_dependencies()
    app.df = pd.DataFrame(
        [
            {
                "drugName": "DrugA",
                "condition": "headache",
                "review": "headache headache migraine relief",
            },
            {
                "drugName": "DrugB",
                "condition": "headache",
                "review": "nausea side effects",
            },
            {
                "drugName": "DrugC",
                "condition": "headache",
                "review": "nausea nausea side effects",
            },
        ]
    )

    recommendations = app.recommend_drugs("headache")

    assert recommendations[0] == "DrugA"


def test_recommendations_return_message_when_condition_has_no_match():
    app = load_app_with_stubbed_dependencies()
    app.df = pd.DataFrame(
        [
            {"drugName": "DrugA", "condition": "headache", "review": "helped my headache"},
        ]
    )

    assert app.recommend_drugs("insomnia") == ["Sorry, no drugs found for the given symptom."]


def test_predict_rejects_missing_symptom():
    app = load_app_with_stubbed_dependencies()
    app.request.get_json = lambda **kwargs: {}

    response, status_code = app.predict()

    assert status_code == 400
    assert response == {"error": "symptom is required"}
