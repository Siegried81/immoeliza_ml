# immoeliza_ml — interactive menu of ML training pipelines (main.py uses input()).
# Run it with -it so the menu can read your keypresses:
#   docker build -t immoeliza-ml .
#   docker run --rm -it immoeliza-ml
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# xgboost wheels need the OpenMP runtime (libgomp1); it is not in python:3.12-slim,
# so `import xgboost` fails at load without this line.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# The pipelines read the cleaned dataset from data/; keep it in the image (copied
# above) or mount a fresh one with -v "$PWD/data:/app/data".
CMD ["python", "main.py"]
