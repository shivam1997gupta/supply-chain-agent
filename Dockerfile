# --- Supply Chain Agent API image ---------------------------------------
# A self-contained image: dependencies, the embedding model, and the seeded
# database are all baked in at build time, so the container needs NO network
# at startup (except your LLM API, which you pass in as an env var at runtime).

FROM python:3.12-slim

WORKDIR /app

# 1. Install Python deps FIRST (this layer is cached and only rebuilds when
#    requirements.txt changes — keeps rebuilds fast while you edit code).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Bake the embedding model INTO the image (~80MB downloaded once, at build
#    time). Without this, every fresh container would download it on first use.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# 3. Copy the application code and scripts.
COPY app ./app
COPY scripts ./scripts

# 4. Generate the SQLite database at build time -> the image is self-contained
#    and reproducible (seed uses a fixed random seed).
RUN python scripts/seed_db.py

# 5. At runtime, never reach out to HuggingFace — use the model baked in step 2.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# 6. The API listens on 8000. IMPORTANT: bind to 0.0.0.0 (not 127.0.0.1) so the
#    container accepts connections from outside itself.
EXPOSE 8000
CMD ["uvicorn", "app.api:api", "--host", "0.0.0.0", "--port", "8000"]
