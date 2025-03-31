# Chatbot Backend

A FastAPI-based backend for a chatbot application.

## Setup

1. Create and activate virtual environment:
```bash
python3.11 -m venv .venv
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



## Running the Application using ngrok
1. Start the server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

2. In a new terminal window, start ngrok:
```
ngrok http 8000
```



## API Endpoints

- `GET /`: Welcome message
- `GET /webhook`: Webhook

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 