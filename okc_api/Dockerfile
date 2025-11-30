FROM python:3.12-slim

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject.toml and install the package and its dependencies
COPY pyproject.toml ./
COPY okc_core/ ./okc_core/
COPY okc_pipeline/ ./okc_pipeline/
COPY okc_api/ ./okc_api/
COPY okc_cli/ ./okc_cli/
# Install package with dev dependencies for testing
RUN pip install --no-cache-dir -e ".[dev]"

RUN python -m spacy download en_core_web_md

ENV PYTHONPATH=/app
CMD ["uvicorn", "okc_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
