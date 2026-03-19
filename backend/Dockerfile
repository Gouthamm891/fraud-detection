# Use the official Python image
FROM python:3.11-slim

# Install uv (fast pip replacement)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency file first (for better caching)
COPY requirements.txt .

# Install dependencies with uv (much faster)
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Train the model at build time
RUN python model/train.py

# Expose the port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]