FROM python:3.12-slim

# Model weights download into /models (compose volume) on first start, not
# into the image: 2-4 GB of weights would bloat every rebuild, and Step 7
# swaps LoRAs without rebuilding.
ENV HF_HOME=/models \
    PYTHONUNBUFFERED=1

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
