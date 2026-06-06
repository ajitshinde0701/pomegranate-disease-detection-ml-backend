# Use a slim Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# Set the working directory
WORKDIR /app

# Install system dependencies needed for OpenCV (cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY anarMitra-ml-backend-api/requirements.txt .

# Install Python dependencies and gunicorn for production
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Copy the rest of the application code from the subdirectory
COPY anarMitra-ml-backend-api/ .

# Ensure the uploads directory exists
RUN mkdir -p uploads

# Expose the port
EXPOSE 5000

# Start the Flask app using Gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 120 app:app"]
