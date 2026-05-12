# Citrus Leaf Disease Detection

This project predicts whether a citrus leaf is healthy or diseased using a trained machine learning model.

## Project Structure

- `backend/`: Flask app, trained model, encoder, and metrics
- `frontend/`: HTML templates and static assets

## Requirements

- Python 3.x
- pip

Install dependencies:

```bash
pip install flask numpy pillow scikit-learn
```

## Run the App

From the `backend` folder:

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Routes

- `/` Home page
- `/prediction` Upload and predict leaf health
- `/about` About page
