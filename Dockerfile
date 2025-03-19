FROM python:3.10.16-slim

# Set the working directory
WORKDIR /sentiment

# Copy the requirements file
COPY dist/sentiments-0.0.1-py3-none-any.whl /sentiment/

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir /sentiment/sentiments-0.0.1-py3-none-any.whl && \
    rm /sentiment/sentiments-0.0.1-py3-none-any.whl
