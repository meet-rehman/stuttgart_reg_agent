FROM python:3.11-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements_optimized.txt .

# Install Python packages with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_optimized.txt && \
    pip cache purge

# Copy application code
COPY . .

# Create directories
RUN mkdir -p memory data government_cache

# Clean up unnecessary files
RUN find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete && \
    find /usr/local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

EXPOSE 8000

CMD ["python", "run_optimized.py"]