# Sentiment Analysis System

A production-ready sentiment analysis system built with **DistilBERT**, **FastAPI**, and **Streamlit**. The system fine-tunes a transformer model on the IMDB movie review dataset and exposes predictions through a REST API and an interactive web UI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Setup Instructions](#3-setup-instructions)
4. [Dataset Information](#4-dataset-information)
5. [Training Instructions](#5-training-instructions)
6. [API Usage](#6-api-usage)
7. [Batch Prediction Usage](#7-batch-prediction-usage)
8. [Docker Usage](#8-docker-usage)
9. [Environment Variables](#9-environment-variables)
10. [Example Requests](#10-example-requests)
11. [Example Responses](#11-example-responses)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Project Overview

### What It Does

This project provides an end-to-end pipeline for binary sentiment classification of movie reviews:

- **Data Preprocessing** — Downloads and tokenizes the IMDB dataset.
- **Model Training** — Fine-tunes `distilbert-base-uncased` for sentiment classification.
- **REST API** — Serves predictions via a FastAPI backend with health checks.
- **Web UI** — Interactive Streamlit dashboard for real-time sentiment analysis.
- **Batch Prediction** — Process CSV files of reviews in bulk.
- **Docker Deployment** — Fully containerised with `docker-compose`.

### Tech Stack

| Layer          | Technology                     |
| -------------- | ------------------------------ |
| Model          | DistilBERT (Hugging Face)      |
| Training       | PyTorch, Transformers, Accelerate |
| API            | FastAPI, Uvicorn               |
| UI             | Streamlit                      |
| Data           | Hugging Face Datasets, Pandas  |
| Containerisation | Docker, Docker Compose       |
| Testing        | Pytest, HTTPX                  |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose                       │
│                                                         │
│  ┌──────────────────┐       ┌────────────────────────┐  │
│  │  sentiment-ui    │       │   sentiment-api        │  │
│  │  (Streamlit)     │──────▶│   (FastAPI + Uvicorn)  │  │
│  │  Port: 8501      │ HTTP  │   Port: 8000           │  │
│  └──────────────────┘       └───────────┬────────────┘  │
│                                         │               │
│                              ┌──────────▼────────────┐  │
│                              │   model_output/       │  │
│                              │   (DistilBERT Model)  │  │
│                              └───────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Components

| Component       | Description                                           |
| --------------- | ----------------------------------------------------- |
| `src/api.py`              | FastAPI application with `/predict` and `/health` endpoints |
| `src/ui.py`               | Streamlit web interface for interactive predictions    |
| `scripts/preprocess.py`   | Data loading and text-cleaning pipeline               |
| `scripts/train.py`        | Model fine-tuning script                              |
| `scripts/batch_predict.py`| Batch prediction utility                              |
| `model_output/`           | Saved model weights and tokenizer                     |
| `data/`                   | Raw and processed dataset files                       |
| `results/`                | Training metrics, evaluation results                  |
| `tests/`                  | Unit and integration tests                            |

---

## 3. Setup Instructions

### Prerequisites

- **Python** 3.11+
- **pip** (latest)
- **Docker** & **Docker Compose** (for containerised deployment)
- **Git**
- **GPU** (optional, recommended for training)

### Local Development Setup

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd "Sentiment Analysis"
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv

   # Linux / macOS
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env as needed
   ```

---

## 4. Dataset Information

### IMDB Movie Review Dataset

| Property       | Value                           |
| -------------- | ------------------------------- |
| Source          | [Hugging Face Datasets](https://huggingface.co/datasets/imdb) |
| Total Reviews  | 50,000                          |
| Training Set   | 25,000 reviews                  |
| Test Set       | 25,000 reviews                  |
| Classes        | Binary — Positive / Negative    |
| Avg. Length     | ~230 words per review           |

The dataset is automatically downloaded and cached by the Hugging Face `datasets` library during preprocessing. Each review is labelled as either **positive** (1) or **negative** (0).

---

## 5. Training Instructions

### Step 1: Preprocess the Data

```bash
python scripts/preprocess.py
```

This downloads the IMDB dataset, cleans the text (removes URLs, HTML tags, special characters), and saves the processed data to `data/processed/`.

### Step 2: Train the Model

```bash
python scripts/train.py
```

Training parameters can be configured via environment variables (see [Environment Variables](#9-environment-variables)):

```bash
# Custom training configuration
LEARNING_RATE=2e-5 BATCH_SIZE=16 NUM_EPOCHS=3 python scripts/train.py
```

The trained model and tokenizer are saved to the `model_output/` directory.

### Step 3: Verify the Model

```bash
pytest tests/
```

---

## 6. API Usage

### Endpoints

| Method | Endpoint   | Description                         |
| ------ | ---------- | ----------------------------------- |
| GET    | `/health`  | Health check — returns service status |
| POST   | `/predict` | Predict sentiment for input text    |

### Request Format (`/predict`)

```json
{
  "text": "This movie was absolutely fantastic! I loved every minute of it."
}
```

### Response Format (`/predict`)

```json
{
  "sentiment": "positive",
  "confidence": 0.9876
}
```

### Starting the API Locally

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The interactive API docs are available at `http://localhost:8000/docs`.

---

## 7. Batch Prediction Usage

Process multiple reviews from a CSV file:

```bash
python scripts/batch_predict.py
```

Input file: `data/unseen/predict_data.csv`  
Output file: `results/predictions.csv`

### Input CSV Format

```csv
text
"This movie was great!"
"Terrible film, waste of time."
"An average movie, nothing special."
```

### Output CSV Format

```csv
text,predicted_sentiment,confidence
"This movie was great!",positive,0.9832
"Terrible film, waste of time.",negative,0.9654
"An average movie, nothing special.",negative,0.5234
```

---

## 8. Docker Usage

### Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up --build -d
```

### Access the Services

| Service | URL                          |
| ------- | ---------------------------- |
| API     | http://localhost:8000        |
| API Docs| http://localhost:8000/docs   |
| UI      | http://localhost:8501        |

### Stop the Services

```bash
docker-compose down
```

### Build Individual Services

```bash
# Build and run API only
docker-compose up --build api

# Build and run UI only (requires API to be healthy)
docker-compose up --build ui
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f ui
```

### Custom Ports

```bash
API_PORT=9000 UI_PORT=9501 docker-compose up --build
```

---

## 9. Environment Variables

| Variable         | Description                              | Default              |
| ---------------- | ---------------------------------------- | -------------------- |
| `API_HOST`       | Host address for the API server          | `0.0.0.0`           |
| `API_PORT`       | Port for the API server                  | `8000`              |
| `UI_PORT`        | Port for the Streamlit UI                | `8501`              |
| `MODEL_PATH`     | Path to the trained model directory      | `/app/model_output` |
| `MODEL_NAME`     | Base model name for training             | `distilbert-base-uncased` |
| `LEARNING_RATE`  | Training learning rate                   | `2e-5`              |
| `BATCH_SIZE`     | Training batch size                      | `16`                |
| `NUM_EPOCHS`     | Number of training epochs                | `3`                 |
| `API_URL`        | API URL used by the UI service           | `http://api:8000`   |

All variables can be set in the `.env` file (see `.env.example` for a template).

---

## 10. Example Requests

### Health Check

```bash
curl -f http://localhost:8000/health
```

### Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic! I loved every minute of it."}'
```

### Negative Sentiment Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible movie. The plot was boring and the acting was awful."}'
```

---

## 11. Example Responses

### Health Check Response

```json
{
  "status": "ok"
}
```

### Positive Sentiment Response

```json
{
  "sentiment": "positive",
  "confidence": 0.9876
}
```

### Negative Sentiment Response

```json
{
  "sentiment": "negative",
  "confidence": 0.9743
}
```

---

## 12. Troubleshooting

### Common Issues

#### 1. Docker build fails with "no space left on device"

```bash
# Clean up unused Docker resources
docker system prune -a
```

#### 2. UI cannot connect to API

- Ensure the API container is **healthy** before starting the UI:
  ```bash
  docker-compose ps
  ```
- Verify `API_URL` is set to `http://api:8000` (Docker internal DNS).
- Check API logs for errors:
  ```bash
  docker-compose logs api
  ```

#### 3. Model not found / MODEL_PATH error

- Ensure `model_output/` contains the trained model files before building the Docker image.
- Train the model first:
  ```bash
  python scripts/train.py
  ```

#### 4. Port already in use

```bash
# Check what's using the port
# Linux / macOS
lsof -i :8000

# Windows
netstat -ano | findstr :8000

# Use a different port
API_PORT=9000 docker-compose up --build
```

#### 5. Out of memory during training

- Reduce the batch size:
  ```bash
  BATCH_SIZE=8 python -m src.train
  ```
- Use CPU if GPU memory is insufficient:
  ```bash
  CUDA_VISIBLE_DEVICES="" python -m src.train
  ```

#### 6. Healthcheck failing / Container restarting

- Check container logs:
  ```bash
  docker-compose logs -f api
  ```
- Increase `start_period` in `docker-compose.yml` if the model takes long to load.
- Verify the `/health` endpoint works locally:
  ```bash
  curl http://localhost:8000/health
  ```

#### 7. Slow first prediction

The first prediction may be slow as the model is loaded into memory. Subsequent predictions will be significantly faster. The healthcheck ensures the model is loaded before the UI starts sending requests.

#### 8. Permission denied errors on Windows

If running on Windows, ensure Docker Desktop has access to the project directory:
- Open **Docker Desktop** → **Settings** → **Resources** → **File sharing**
- Add the project directory to the shared drives list.

---

## License

This project is for educational and demonstration purposes.
