# Hybrid Recommender System with Explainable Product Search

## Overview

This project presents a hybrid recommender system for e-commerce product discovery, combining semantic retrieval and collaborative filtering with an explainability layer.

The system is designed to:
- Improve product relevance through semantic search
- Personalize recommendations using user behavior (ALS)
- Transparency via human-readable explanations for ranked results

---
## Motivation
Traditional recommender systems typically rely on either:

Content-based approaches (semantic similarity)
Collaborative filtering (user–item interactions)

These approaches suffer from key limitations:

- Content-based methods lack personalization
- Collaborative filtering struggles with cold-start scenarios
- Both approaches provide limited transparency in ranking decisions

This project addresses these challenges by combining retrieval, personalization, and explainability into a unified system.

---

## Key Features

-  **Semantic Search**
  - Transformer-based embeddings for query understanding
  - Cosine similarity for product matching

-  **Collaborative Filtering (ALS)**
  - Learns user-item interaction patterns
  - Learns latent preference patterns
  - Provides personalized ranking signals

-  **Hybrid Ranking**
  - Combines semantic relevance + ALS scores
  - Robust to cold-start scenarios
  - Produces a unified ranking that balances relevance and personalization

-  **Explainability Layer**
  - Generates structured explanations using:
    - Ratings
    - Review counts
    - Verified purchase ratio
    - Product features

-  **LLM Enhancement**
  - TinyLlama used for polishing explanations
  - Improves readability and natural language quality

---

## Pipeline Breakdown

### 1. Data Processing
- Product metadata and review data cleaned and structured
- Data loaded using pandas DataFrames (tabular structure)
- Merging datasets for enriched signals (ratings, reviews, etc.)

### 2. Semantic Retrieval
- Encoded product text into embeddings (Product text → embeddings)
- Converted user queries into vector representations (Query → embedding)
- Retrieved top-k similar products (Similarity search → top-k candidates)

### 3. Collaborative Filtering (ALS)
- User-item interactions modeled
- Generates personalized recommendations
- Handles known-user scenarios

### 4. Hybrid Ranking
- Combines:
  - semantic_score
  - als_score
- Produces final_ranking_score

### 5. Explanation Layer
- Feature extraction from product titles/descriptions
- Trust signals:
  - average rating
  - review count
  - verified purchase ratio
- Generates:
  - Overview summary
  - Best product reasoning
  - Quick recommendation categories

---

## Example Output

### Query
wireless headphones for studying


### Overview
For 'wireless headphones for studying', the strongest results emphasize wireless, microphone, over-ear, and portability.

The top-ranked products combine strong query relevance with personalized ranking signals, while user feedback helps highlight the most reliable options.


### Best Overall Explanation
Wireless Kids Headphones with Microphones ranks first because it combines strong feature relevance with high engagement and consistent user feedback.


---

## Technologies Used

- Python
- PySpark
- Pandas
- NumPy
- Scikit-learn
- Hugging Face Transformers
- ALS (Spark MLlib)
- TinyLlama 
- Hadoop / HDFS

---

## Current Status

 Semantic retrieval  
 ALS model  
 Hybrid ranking  
 Explanation layer  
 API integration (in progress)  
 Web interface (planned)  

---

## Future Improvements

- Real-time API deployment (FastAPI)
- Frontend interface for live search
- Better ranking calibration
- Advanced LLM-based explanations
- Multimodal search (image + text)

---

## Author

Preye Nabena  
MSc Artificial Intelligence & Machine Learning  

---
## Research Direction
This project contributes to research in:

- Recommender Systems
- Natural Language Processing
- Explainable AI
- Intelligent Decision Support Systems

---

## License

This project is for academic and research purposes.
