# Dockerfile

FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main.py file into the container
COPY main.py .

# Expose the port for the uvicorn server
EXPOSE 8000

# Command to run the uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]