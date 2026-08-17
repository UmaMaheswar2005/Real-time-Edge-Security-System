FROM python:3.12-slim

WORKDIR /app

# Install C-libraries required by OpenCV on Linux
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into container
COPY . .

# Expose container port
EXPOSE 8000

# Bind dynamically to $PORT provided by Render (defaults to 8000)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]