# Running Pipeline Performance Tests

This directory contains comprehensive tests for the OKC pipeline, including unit tests for individual stages and end-to-end pipeline tests.

## Prerequisites

1. **Install dependencies** (including dev dependencies):
   ```bash
   pip install -e ".[dev]"
   ```

2. **Download spaCy model** (required for entity extraction and sentence splitting):
   ```bash
   python -m spacy download en_core_web_md
   ```

3. **Ensure sentence-transformers model is available** (will be downloaded automatically on first use):
   - The tests use `sentence-transformers/all-MiniLM-L6-v2` which will be downloaded automatically

## Running Tests

### Run All Tests

```bash
# From the project root directory
pytest tests/

# Or with verbose output
pytest tests/ -v

# Or with even more detail
pytest tests/ -vv
```

### Run Specific Test Files

```bash
# Run only chunking tests
pytest tests/test_stage_chunking.py

# Run only entity extraction tests
pytest tests/test_stage_entities.py

# Run only end-to-end pipeline tests
pytest tests/test_pipeline_e2e.py

# Run only metrics tests
pytest tests/test_metrics.py
```

### Run Specific Test Functions

```bash
# Run a specific test function
pytest tests/test_stage_chunking.py::test_chunking_basic

# Run tests matching a pattern
pytest tests/ -k "entity"

# Run tests matching multiple patterns
pytest tests/ -k "entity or claim"
```

### Run Tests by Category

```bash
# Run only unit tests (individual stages)
pytest tests/test_stage_*.py

# Run only integration tests
pytest tests/test_pipeline_e2e.py tests/test_metrics.py
```

## Running Tests in Docker Container

If you're running tests from within a Docker container:

### Option 1: Install httpx in running container (quick fix)
```bash
docker exec okc_api pip install httpx
docker exec okc_api pytest tests/
```

### Option 2: Rebuild container with updated dependencies (recommended)
```bash
# Rebuild the container to include httpx and dev dependencies
docker-compose build
docker exec okc_api pytest tests/
```

The Dockerfile has been updated to install dev dependencies including `httpx` and `pytest`.

## Test Output Options

### Verbose Output
```bash
pytest tests/ -v          # Show test names
pytest tests/ -vv          # Show test names and assertions
pytest tests/ -s           # Show print statements
```

### Show Coverage
```bash
# Install pytest-cov first: pip install pytest-cov
pytest tests/ --cov=okc_pipeline --cov=okc_core --cov-report=html
```

### Stop on First Failure
```bash
pytest tests/ -x
```

### Run Tests in Parallel (if pytest-xdist is installed)
```bash
# Install: pip install pytest-xdist
pytest tests/ -n auto
```

## Understanding Test Results

### Test Structure

- **`test_stage_chunking.py`**: Tests for document chunking
- **`test_stage_sentences.py`**: Tests for sentence splitting
- **`test_stage_entities.py`**: Tests for entity extraction with accuracy metrics
- **`test_stage_claims.py`**: Tests for claim detection with accuracy metrics
- **`test_stage_relations.py`**: Tests for relation extraction with accuracy metrics
- **`test_pipeline_e2e.py`**: End-to-end pipeline tests
- **`test_metrics.py`**: Metrics aggregation and reporting tests

### Metrics Output

The tests calculate and report:
- **Precision**: How many of the extracted items are correct
- **Recall**: How many of the expected items were found
- **F1 Score**: Harmonic mean of precision and recall

These metrics are broken down by:
- Entity type (Person, Organization, TechnicalArtifact, etc.)
- Relation type (is_a, improves, evaluated_on, etc.)

## Troubleshooting

### Database Issues

The tests use SQLite in-memory database (`test.db`). If you see database errors:
- Make sure the test database file can be created in the current directory
- Check that SQLAlchemy models are properly imported

### spaCy Model Not Found

If you see errors about missing spaCy models:
```bash
python -m spacy download en_core_web_md
```

Or in Docker:
```bash
docker exec okc_api python -m spacy download en_core_web_md
```

### Import Errors

If you see import errors, make sure you're running from the project root:
```bash
# From project root
pytest tests/
```

Or install the package in development mode:
```bash
pip install -e .
```

### Missing httpx Error

If you see `RuntimeError: The starlette.testclient module requires the httpx package`:
- Install httpx: `pip install httpx`
- Or rebuild Docker container: `docker-compose build`
- Or install in running container: `docker exec okc_api pip install httpx`

### Slow Tests

Some tests may be slow because they:
- Download models on first run
- Process text with spaCy
- Generate embeddings

This is normal. Subsequent runs will be faster due to caching.

## Example Test Run

```bash
# Run all tests with verbose output
$ pytest tests/ -v

# Output will look like:
# ============================= test session starts ==============================
# platform linux -- Python 3.10.x, pytest-7.x.x
# collected 50 items
#
# tests/test_stage_chunking.py::test_chunking_basic PASSED
# tests/test_stage_chunking.py::test_chunking_respects_sentence_boundaries PASSED
# tests/test_stage_entities.py::test_entity_extraction_accuracy PASSED
# ...
# ============================= 50 passed in 45.23s ==============================
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. They:
- Use isolated test databases (SQLite)
- Don't require external services
- Are deterministic (same inputs produce same outputs)
- Can be run in parallel

Example CI configuration:
```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pip install -e ".[dev]"
    python -m spacy download en_core_web_md
    pytest tests/ -v
```
