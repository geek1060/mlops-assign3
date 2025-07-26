FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir scikit-learn joblib

CMD ["python", "src/predict.py"]
