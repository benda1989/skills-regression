FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV ENV=production

EXPOSE 8000

CMD ["uvicorn", "regression_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
