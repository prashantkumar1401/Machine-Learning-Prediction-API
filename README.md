# Machine Learning Prediction API

A small machine-learning service that trains a scikit-learn classification model and exposes predictions through a FastAPI REST API.

## Features
- Scikit-learn classification model
- FastAPI REST endpoints
- Pydantic request validation
- Interactive Swagger documentation
- Health-check endpoint
- Automated API tests

## Architecture

```text
Synthetic training data
        ↓
Scikit-learn model
        ↓
FastAPI endpoint
        ↓
Validated JSON request
        ↓
Prediction + confidence
```

## Tech Stack

Python • scikit-learn • FastAPI • Pydantic • Uvicorn • pytest

## API Endpoints

### GET `/health`

Returns service status.

### POST `/predict`

Example request:

```json
{
  "age": 35,
  "income": 50000,
  "tenure": 4
}
```

Example response:

```json
{
  "prediction": "lower-risk",
  "confidence": 0.72
}
```

## Run Locally

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

Run tests:

```bash
pytest
```

## Scope Note

The included dataset is synthetic and intentionally small for learning purposes. The resulting model is **not** a validated financial, credit, medical, or production decision system.

## Future Improvements

- Replace synthetic data with a documented real dataset
- Add model persistence with joblib
- Add preprocessing pipelines
- Add model evaluation metrics
- Add Docker
- Add CI with GitHub Actions
- Add structured logging
- Add API authentication and rate limiting

## Portfolio Skills

Machine learning • model serving • FastAPI • REST APIs • Pydantic • data validation • testing • Python • scikit-learn

## Author

**Prashant Kumar**

GitHub: https://github.com/prashantkumar1401
