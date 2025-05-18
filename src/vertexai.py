from google.cloud import aiplatform

# This script is intended to be run by YOU to submit the job.
# It will NOT run inside the Vertex AI training container.

aiplatform.init(project="gemma-garage", 
                location="us-central1",
                staging_bucket="gs://gemma-garage-datasets") # Staging bucket for Vertex AI internals

job = aiplatform.CustomContainerTrainingJob(
    display_name="gemma-peft-finetune-job", # Changed display name slightly for clarity
    container_uri="gcr.io/gemma-garage/gemma-finetune:latest", # Built by your Dockerfile & Cloud Build
    # model_serving_container_image_uri is for deploying the trained model, not for training itself.
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest", # Optional
)

# Arguments for your training_task.py script
training_args = [
    "--dataset=gs://gemma-garage-datasets/questionsjson", # GCS path to your data
    "--output_dir=gs://gemma-garage-models/gemma-peft",    # GCS path for output
    "--model_name=google/gemma-3-1b-it", # Or your desired model
    "--epochs=1",                   # Example
    "--learning_rate=0.0002",       # Example
    "--lora_rank=4"                 # Example
    # Add other arguments as needed by training_task.py
]

# Define the machine type and accelerators for the training job
job.run(
    base_output_dir="gs://gemma-garage-models/gemma-peft-vertex-output", # Vertex AI specific outputs
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=training_args,
    replica_count=1,
    # service_account="YOUR_VERTEX_AI_CUSTOM_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com" # Optional: if needed
)

print("Vertex AI Training Job submitted.")
