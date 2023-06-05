# Use slim Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install necessary libraries, add new libraries as a new line to take advantage of caching
RUN pip install scikit-learn
RUN pip install uvicorn 
RUN pip install fastapi 
RUN pip install pydantic 

# Copy the app code into the container
COPY main.py .
COPY models/ models/

# Expose the port that the app will run on
EXPOSE 8000

# Start the app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
