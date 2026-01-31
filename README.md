# Predicting Protein Secondary Structure Using Machine Learning

## Project Overview
This project explores the use of supervised machine learning techniques to predict protein secondary structure (coil, helix, and sheet) from amino acid composition features. The goal is to evaluate how classical ML models perform on biological sequence data and to compare their strengths using domain-appropriate evaluation metrics.

## Problem Statement
Protein secondary structure plays a critical role in understanding protein function and biological processes. Traditional experimental approaches are costly and time-consuming, motivating the need for computational prediction methods.

## Methodology

### Data Preprocessing & Feature Engineering
- Protein sequences were converted into amino acid composition (AAC) feature vectors  
- Features were normalized and standardized prior to model training  
- Data preprocessing ensured consistency and suitability for supervised learning  

### Machine Learning Models
The following classifiers were trained and evaluated:
- Logistic Regression  
- Random Forest  

Model selection was based on comparative performance across multiple evaluation metrics.

## Model Evaluation
Models were evaluated using:
- Confusion Matrices (class-wise performance)  
- One-vs-Rest ROC Curves  
- AUC scores for each secondary structure class  

Visualizations were generated to interpret class imbalance, discriminative power, and model robustness.

## Tools & Technologies
- Python  
- scikit-learn  
- Pandas  
- NumPy  
- VS Code  

## What This Project Demonstrates
- Application of machine learning to biological sequence data  
- Feature engineering from raw biological inputs  
- Comparative evaluation of supervised learning models  
- Translation of domain-specific problems into ML pipelines  
- Strong foundation in bioinformatics-focused data science

