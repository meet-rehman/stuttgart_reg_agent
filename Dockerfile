FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements_optimized.txt .
RUN pip install --no-cache-dir -r requirements_optimized.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p memory data government_cache

# Expose port (Railway will set PORT environment variable)
EXPOSE 8000

# Run the application
CMD ["python", "run_optimized.py"]