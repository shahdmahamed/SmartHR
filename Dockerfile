# Use Python 3.10
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files
COPY backend/ ./backend/

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "backend.ML.app:app", "--host", "0.0.0.0", "--port", "8000"]

#docker build -t smart-hr-api .
#docker run -d -p 8000:8000 smart-hr-api
#http://localhost:8000/docs

