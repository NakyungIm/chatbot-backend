# Chatbot Backend

A FastAPI-based backend for a chatbot application.

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip3 install -r requirements.txt
```


## Running the Application

1. Start the server:
```bash
fastapi dev main.py
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

- `GET /`: Welcome message
- `GET /webhook`: Webhook

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 