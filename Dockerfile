FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (Hugging Face requirement)
RUN useradd -m -u 1000 user
USER user

# Copy requirements and install dependencies
# We use --user so it installs to /home/user/.local/bin
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Ensure the local bin is on the PATH for gunicorn
ENV PATH="/home/user/.local/bin:${PATH}"

# Copy all the application files
COPY --chown=user:user . /app

# Ensure runtime directories exist and are writable
RUN mkdir -p /app/sessions /app/uploads /app/saved_models && \
    chmod 777 /app/sessions /app/uploads /app/saved_models

# Expose port 7860 (Hugging Face Default)
EXPOSE 7860

# Command to run the application using gunicorn on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
