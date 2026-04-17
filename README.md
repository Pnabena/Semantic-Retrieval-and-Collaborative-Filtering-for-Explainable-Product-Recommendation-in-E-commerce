# Integrating Semantic Retrieval and Collaborative Filtering for Explainable Product Recommendation in E-commerce

## Overview
This project presents the design and evaluation of a hybrid recommender system that integrates semantic retrieval with collaborative filtering to improve both recommendation quality and interpretability in e-commerce platforms.

The system combines transformer-based text embeddings with user–item interaction modelling to support personalized, context-aware, and explainable product recommendations, while remaining robust in cold-start scenarios.

## Motivation
Traditional recommender systems typically rely on either:

Content-based methods (semantic similarity)
Collaborative filtering (user–item interactions)

Each approach has limitations:

Content-based methods struggle with personalization
Collaborative filtering struggles with cold-start users/items
Both approaches often lack transparency in ranking decisions

This project addresses these challenges by integrating:

Semantic understanding
User behaviour modelling
Explainability

## System Architecture
The proposed system consists of four main components:

1. Semantic Retrieval Layer
Uses transformer-based embeddings (e.g., Sentence Transformers)
Encodes product descriptions into dense vector representations
Supports semantic search based on user queries
2. Collaborative Filtering Layer
Implements Alternating Least Squares (ALS)
Learns latent user–item interaction patterns
Captures implicit preference signals from historical data
3. Hybrid Ranking Module
Combines:
Semantic similarity scores
Collaborative filtering scores
Produces a unified ranking of candidate products
Balances relevance and personalization
4. Explanation Layer
Generates human-readable recommendation justifications
Uses:
Product metadata
Ratings and review signals
Feature-based reasoning

Example Output:

“This item is recommended because it is similar in style to your search and is highly rated by users with similar preferences.”

## Dataset
Large-scale e-commerce dataset (product descriptions, user interactions, ratings)
Includes:
Product metadata
User–item interactions
Review-based signals
Amazon 23' large dataset (Electronics)

## Research Direction
This project contributes to ongoing research in:

Recommender Systems
Natural Language Processing
Explainable AI
Intelligent Decision Support Systems
