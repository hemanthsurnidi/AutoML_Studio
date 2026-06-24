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

# Ensure the local bin is on the PATH for gunicorn
ENV PATH="/home/user/.local/bin:${PATH}"

# Copy requirements and install dependencies
# We use --user so it installs to /home/user/.local/bin
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy all the application files
COPY --chown=user:user . /app

# Switch to root temporarily to create directories and fix permissions
USER root
RUN mkdir -p /app/sessions /app/uploads /app/saved_models && \
    chown -R user:user /app/sessions /app/uploads /app/saved_models && \
    chmod -R 777 /app/sessions /app/uploads /app/saved_models

# Switch back to the required non-root user
USER user

# Expose port 7860 (Hugging Face Default)
EXPOSE 7860

# Command to run the application using gunicorn on port 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
