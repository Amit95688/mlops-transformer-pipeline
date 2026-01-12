# Multi-stage build: builder stage
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Set environment variables
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy project files
COPY config.py .
COPY dataset.py .
COPY model.py .
COPY train.py .
COPY tokenizer_en.json .
COPY tokenizer_hi.json .

# Create directories for outputs
RUN mkdir -p models runs mlruns

# Expose MLflow UI port (optional, for remote access)
EXPOSE 5000

# Default command: training
CMD ["python", "train.py"]
