# CIvolunteer Matching System
A Django-based backend system for matching volunteers to tasks. The system uses standardized ESCO skills, embedding-based skill extraction, and overlap-based recommendation strategies.

## Features

- ESCO skill extraction from unstructured text using embedding and LLM-based methods.
- Standardized volunteer and task profiles.
- Overlap-based recommendation engine using standardized ESCO skills
- REST API for user and task data ingestion
- Scheduled and event-driven matching pipeline (via Django signals and management commands).
- PostgreSQL with pgvector for fast similarity search.
- Configurable skill extraction and matching components.

## Tech Stack
- Python, Django
- PostgreSQL + pgvector
- Hugging Face Transformers (for embeddings)
- SentenceTransformers
- REST framework (DRF)
- FLAN-T5 LLM fallback for skill extraction

## Project Structure
- recommender_rest: REST endpoints for profiles
- recommender_test: UI for method evaluation
- recommender_profile: User and Task profile models
- recommender_kb: Knowledge Base models & management
- recommender_core: Matching logic, extractors, matchers
- recommender: Settings

## Setup
### Requirements
- Python 3.11
- Django 4.2
- PostgreSQL with pgvector extension

### Installation

```bash
# clone the repository
git clone https://github.com/abood1987/recommender.git

# setup virtualenv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# setup database
createdb matching_db
# ensure pgvector extension is installed
# run migrations
python manage.py migrate

# create superuser
python manage.py createsuperuser

# running the Server
python manage.py runserver
```

## Skill Extraction & Embedding
Embeddings generated from ESCO skill labels.
Primary extraction: Cosine similarity matching with embedding model (esco-context-skill-extraction).
Fallback: Fine-tuned FLAN-T5 LLM.

## Matching Algorithm
Default: Overlap matcher with support for broader skill inclusion.
Other methods (TF-IDF, Fuzzy, Embedding Cosine) available for evaluation

## Evaluation & Experimentation
#### Colab Notebooks:
data/colab/*.ipynb

#### Evaluation metrics include:
Precision, Recall, F1-Score (for extraction)
Avg Matches/User, Unique Jobs, Matched Users/Jobs (for matching)

## Matching Process
Triggered by post_save signals on profile updates.

## Generate recommendations
Can be executed via daily scheduled management command:
````bash
python manage.py generate_recommendations
````

## Testing Interface
A simple test frontend (recommender_test) allows UI-based exploration of extraction/matching strategies.

