# Intelligent Systems Framework for E-Commerce Innovation

## Overview

This project presents a dual AI-driven framework designed to enhance product discovery and metadata quality in large-scale e-commerce platforms. 

The system addresses two core challenges:

1. Improving product search accuracy for natural-language and intent-based queries.
2. Automatically generating structured and SEO-optimized product tags using image-based classification and language models.

The framework integrates Information Retrieval, Natural Language Processing, Computer Vision, and Large Language Models into a unified architecture.

---

## Problem Statement

Modern e-commerce platforms face two major limitations:

- Traditional keyword-based search engines fail to interpret conversational and goal-based queries.
- Product catalogs often contain incomplete or inconsistent metadata, reducing discoverability.

This project proposes a hybrid AI solution to overcome these limitations through semantic retrieval and automated tagging.

---

## System Architecture

The framework consists of two major components:

### 1. Advanced Hybrid Product Search Engine

A semantic-aware retrieval system combining:

- BM25 lexical search
- SentenceTransformer embeddings (all-MiniLM-L6-v2)
- FAISS vector similarity search (HNSW indexing)
- Query expansion using WordNet synonyms
- Rule-based intent detection
- Category alignment and filtering
- Hybrid score fusion (70% semantic, 30% lexical)
- LLM-based reasoning layer for explainability
- Not-found detection module
- Maximal Marginal Relevance (MMR) for result diversification

This architecture allows the system to interpret abstract, conversational, and goal-oriented queries effectively.

---

### 2. Hybrid Product Tagging Pipeline (CNN + LLM)

An automated metadata generation pipeline using:

- ResNet50V2 (transfer learning, ImageNet pre-trained)
- Two-phase fine-tuning strategy
- Multi-label classification
- Hierarchical tag completion logic
- Threshold optimization (0.15 cutoff)
- LLM-based SEO tag enhancement

The system classifies product images into categories and generates structured tags to improve catalog completeness and search visibility.

---

## Dataset

Source: Flipkart E-Commerce Sample Dataset (Kaggle)

- ~20,000 product records
- 15 structured attributes
- 6,400+ unique hierarchical category paths

Key attributes used:
- Product name
- Category tree
- Description
- Specifications
- Price data
- Image URLs

Data preprocessing included:
- Text normalization
- Category hierarchy simplification
- Creation of unified `search_text`
- Datetime conversion
- Null value handling
- Removal of noisy attributes

---

## Hybrid Search Methodology

### Step 1: Data Preparation
- Concatenated product metadata into a unified `search_text`
- Extracted top-level category (`main_category`)

### Step 2: Lexical Retrieval
- BM25 index over search_text
- Default hyperparameters (k1, b)

### Step 3: Semantic Retrieval
- 384-dimensional embeddings via SentenceTransformer
- Indexed using FAISS HNSW
- Approximate nearest-neighbor retrieval

### Step 4: Score Normalization
- BM25: Z-score normalization
- Cosine similarity: Min-max scaling

Final score formula:

Final Score = 0.7 × Semantic Score + 0.3 × BM25 Score

### Step 5: Intent Detection
- Rule-based activity recognition
- Category down-weighting for mismatches

### Step 6: Query Expansion
- WordNet synonym enrichment
- Lower-weight merging of expanded results

### Step 7: LLM Reasoning
- Product explanations
- Context-aware summaries
- Justification of ranking

### Step 8: Not-Found Logic
- Threshold-based rejection of irrelevant results

---

## Example Query Results

Query: "I need comfortable running shoes for daily jogging"
- N Five Running Shoes — 94% match
- JQR Sports Running Shoes — 80% match

Query: "I am going for swimming"
- Novicz Swimming Goggles — 81%
- Speedo Aquashort — 65%

Query: "My school is reopening"
- Angel Glitter School Bag — 58%
- School Shoes — 72%

Query: "Food items"
- System returns contextual "Not Found" message

---

## Hybrid Tagging Model Performance

Raw Validation Performance:
- Precision: 73.5%
- Recall: 76.3%
- F1 Score: 74.9%
- Validation Loss: 0.0011

Threshold-Optimized Performance:
- Micro F1 Score: 0.7289
- Macro F1 Score: 0.0687

Insights:
- Strong performance on common, high-level categories
- Lower performance on rare and fine-grained tags
- Hierarchical post-processing ensures logical tag completion

---

## Technologies Used

- Python
- Scikit-learn
- Rank-BM25
- SentenceTransformers
- FAISS (HNSW)
- TensorFlow / Keras
- ResNet50V2
- WordNet
- Large Language Models (Mistral-based reasoning)
- Pandas / NumPy

---

## Key Contributions

- Hybrid semantic + lexical search architecture
- Intent-aware retrieval system
- Diversified ranking via MMR
- CNN-based hierarchical multi-label classification
- LLM-powered metadata generation
- Unified scalable microservices-inspired design

---

## Limitations

- Dataset imbalance impacts fine-grained classification
- Rule-based intent detection lacks deep contextual modeling
- Pre-trained embeddings not fine-tuned to domain
- LLM tagging increases latency
- Offline evaluation without precision@k benchmarking

---

## Future Improvements

- Transformer-based intent classification
- Domain-specific embedding fine-tuning
- Learning-to-rank frameworks
- Hierarchical multi-task classification
- Latency optimization via model compression
- A/B testing in production environments

---

## Conclusion

This project delivers a scalable AI-driven e-commerce intelligence framework that enhances both:

- User experience through semantic-aware product discovery
- Seller optimization through automated, structured metadata generation

By integrating NLP, computer vision, and large language models into a unified architecture, the system moves beyond traditional keyword search toward a future-ready, context-aware e-commerce ecosystem.

---

## Authors

Group 3  
M.Sc. Statistics and Data Science  
SVKM’s NMIMS (2024–2026)

- Himani Grover
- Khushi Gupta
- Sneha Maheshwari
- Ritesh Patil
- Prashant Srivastava

Project Supervisor: Mr. Rutik Bhurke

---

## License

This project is intended for academic and research purposes.
