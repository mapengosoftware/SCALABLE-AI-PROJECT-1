# Scalable ML Project Template

This template helps you:
- Load and process data
- Train, evaluate, and save models
- Serve predictions via FastAPI
- Deploy using Docker

## Usage

```bash
# Train model
python src/train.py

# Evaluate model
python src/evaluate.py

# Start API server
uvicorn api.main:app --reload
```

## Docker
```bash
docker build -t ml-api .
docker run -p 8000:8000 ml-api
```
Or Run in Local server 
on main.py folder command uvicorn main:app --reload  load on browser http://127.0.0.1:8000/docs 

## Requirements
```bash
pip install -r requirements.txt
```
