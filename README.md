# ML FastAPI Docker

A containerized Machine Learning prediction API built with FastAPI, Scikit-learn and Docker.

## Project Overview

This project demonstrates how a trained Machine Learning model can be served through a FastAPI REST API and packaged into a Docker container.

## Architecture

Client
↓
Docker Container
↓
FastAPI API
↓
Pydantic Validation
↓
Preprocessing Pipeline
↓
ML Model
↓
Prediction
↓
JSON Response

## Features

- FastAPI REST API
- Pydantic validation
- Scikit-learn ML model
- Preprocessing pipeline
- Model artifact loading
- Health endpoint
- Readiness endpoint
- Single prediction
- Batch prediction
- Dockerfile
- Docker image
- Docker container
- Port mapping
- Environment variables
- Docker Compose
- Docker Hub
- Swagger/OpenAPI
- Docker logs and debugging

## API Endpoints

### GET /health

Checks whether the API is running.

### GET /ready

Checks whether the ML model is loaded.

### POST /predict

Predict for one employee.

Example:

```json
{
  "age": 30,
  "salary": 50000
}
