# Use a Python image with good compatibility for dlib
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libboost-all-dev \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy your project files into the container
COPY . /app

# Copy requirements.txt (this allows Docker to install dependencies)
COPY requirements.txt .

# Install Python dependencies from the requirements.txt file
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Optionally, run migrations and start the Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
