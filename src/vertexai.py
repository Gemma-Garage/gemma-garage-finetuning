from google.cloud import aiplatform

# This script is intended to be run by YOU to submit the job.
# It will NOT run inside the Vertex AI training container.

# Ensure these bucket names are created in your 'llm-garage' project
NEW_STAGING_BUCKET = "gs://llm-garage-vertex-staging" # Example, choose a unique name
NEW_DATA_BUCKET = "gs://llm-garage-datasets"         # Example, choose a unique name
NEW_MODEL_OUTPUT_BUCKET = "gs://llm-garage-models/gemma-peft-vertex-output" #"gs://llm-garage-models"   # Example, choose a unique name

aiplatform.init(project="llm-garage", 
                location="us-central1",
                staging_bucket=NEW_STAGING_BUCKET)

job = aiplatform.CustomContainerTrainingJob(
    display_name="gemma-peft-finetune-job-llm-garage",
    container_uri="gcr.io/llm-garage/gemma-finetune:latest", # Image from llm-garage GCR
    # model_serving_container_image_uri is for deploying the trained model, not for training itself.
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest", # Optional
)

# Arguments for your training_task.py script
training_args = [
    f"--dataset={NEW_DATA_BUCKET}/questions.json",         # GCS path to your data in llm-garage
    f"--output_dir={NEW_MODEL_OUTPUT_BUCKET}/model/", # GCS path for output in llm-garage
    "--model_name=google/gemma-3-1b-it", # Or your desired model, updated to a valid gemma model
    "--epochs=1",                   # Example
    "--learning_rate=0.0002",       # Example
    "--lora_rank=4"                 # Example
    # Add other arguments as needed by training_task.py
]

# Define the machine type and accelerators for the training job
job.run(
    base_output_dir=f"{NEW_MODEL_OUTPUT_BUCKET}", # Vertex AI specific outputs in llm-garage
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=training_args,
    replica_count=1,
    # service_account="YOUR_VERTEX_AI_CUSTOM_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com" # Optional: if needed
)

print("Vertex AI Training Job submitted.")
