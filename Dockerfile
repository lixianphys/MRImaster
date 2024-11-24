# Use a base image with Python installed
FROM python:3.8-slim

# Set a working directory in the container
WORKDIR /app

# Copy only the pyproject.toml and poetry.lock files to leverage Docker caching
COPY pyproject.toml poetry.lock /app/

# Install Poetry
RUN pip install poetry

# Install Streamlit
RUN pip install streamlit

# Install dependencies without creating a virtual environment
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi

# Copy the rest of the app code into the container
COPY . /app

# Expose the port that Streamlit will run on
EXPOSE 8501

# Define the command to run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
