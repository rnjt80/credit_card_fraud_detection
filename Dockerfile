FROM python:3.9-slim AS builder

WORKDIR /app

COPY flask_app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY flask_app /app/
COPY models/power_transformer.pkl /app/models/power_transformer.pkl

FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app /app

EXPOSE 8000

#local
CMD ["python", "app.py"]
