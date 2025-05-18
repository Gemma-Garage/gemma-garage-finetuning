FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the finetuning logic and the new training task script
COPY src/finetuning.py .
COPY src/training_task.py . 
# The vertexai.py (job submission script) is NOT needed inside this container

# Set the Python path if your scripts import each other using relative paths
# and are in subdirectories (not the case here as they are copied to /app)
# ENV PYTHONPATH=/app

# This is the script Vertex AI will execute inside the container.
# The `args` from your job submission (vertexai.py) will be passed to this script.
ENTRYPOINT ["python", "training_task.py"]
