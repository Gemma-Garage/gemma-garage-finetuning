FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Declare the Hugging Face token build argument
ARG HF_TOKEN

# Set the Hugging Face token as an environment variable
ENV HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

# Install build-essential for C compilers
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN huggingface-cli login --token ${HF_TOKEN}  

# Copy the finetuning logic and the new training task script
COPY src/finetuning.py .
COPY src/training_task.py . 
COPY src/main.py .
# The vertexai.py (job submission script) is NOT needed inside this container

# Set the Python path if your scripts import each other using relative paths
# and are in subdirectories (not the case here as they are copied to /app)
# ENV PYTHONPATH=/app

# This is the script Vertex AI will execute inside the container.
# The `args` from your job submission (vertexai.py) will be passed to this script.
#ENTRYPOINT ["python", "training_task.py"]

EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
