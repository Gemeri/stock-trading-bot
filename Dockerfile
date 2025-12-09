FROM python:3.12.8

# Set working directory
WORKDIR /

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Run forest.py
CMD ["python", "forest.py"]
