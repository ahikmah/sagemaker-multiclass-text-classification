# Building and Deploying a Multiclass Text Classification Model with DistilBERT on AWS SageMaker

## Overview
Welcome to my project on building and deploying a machine learning model using AWS SageMaker! In this project, I explored the capabilities of the Hugging Face `transformers` library and the `DistilBERT` model to perform text classification on the [News Aggregator Dataset](https://archive.ics.uci.edu/dataset/359/news+aggregator).

## Project Highlights
- **Model**: I chose `DistilBERT`, a streamlined and efficient version of BERT, perfect for text classification tasks.
- **Dataset**: The [News Aggregator Dataset](https://archive.ics.uci.edu/dataset/359/news+aggregator) provided a rich collection of news articles, categorized into various topics.
- **Platform**: AWS SageMaker was my go-to for its powerful model training and deployment features.
- **Libraries**: Hugging Face `transformers` and `datasets` libraries were instrumental in implementing and handling the model and data.
- **Serverless**: AWS Lambda was used to create a serverless architecture for handling inference requests.

## Objectives
My main goals for this project were to:
- Set up an AWS SageMaker environment from scratch.
- Preprocess text data to prepare it for model training.
- Train a transformer-based model on a substantial dataset.
- Deploy the trained model to an endpoint for real-time inference.
- Utilize AWS Lambda to handle inference requests in a serverless manner.

## Workflow
Here's a brief rundown of the steps I followed:
1. **Setup**: Configured the AWS environment and installed the necessary libraries.
2. **Data Preparation**: Downloaded and preprocessed the dataset to make it suitable for model training.
3. **Model Training**: Trained the `DistilBERT` model using SageMaker's managed training services.
4. **Model Deployment**: Deployed the trained model to a SageMaker endpoint for real-time predictions.
5. **Serverless Inference**: Used AWS Lambda to handle inference requests, providing a scalable and cost-effective solution.

## Results
The project was a success! I managed to deploy a text classification model that accurately categorizes news articles.