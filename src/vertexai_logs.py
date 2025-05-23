import time # Added for polling
from google.cloud import aiplatform, logging as cloud_logging
from google.cloud import logging_v2
from google.cloud.aiplatform import CustomJob
#from google.cloud.aiplatform.gapic.schema.trainingjob import definition_v1
from datetime import datetime, timezone
import threading


# This script is intended to be run by YOU to submit the job.
# It will NOT run inside the Vertex AI training container.

# Ensure these bucket names are created in your 'llm-garage' project
NEW_STAGING_BUCKET = "gs://llm-garage-vertex-staging" # Example, choose a unique name
NEW_DATA_BUCKET = "gs://llm-garage-datasets"         # Example, choose a unique name
NEW_MODEL_OUTPUT_BUCKET = "gs://llm-garage-models/gemma-peft-vertex-output" #"gs://llm-garage-models"   # Example, choose a unique name

aiplatform.init(project="llm-garage", 
                location="us-central1",
                staging_bucket=NEW_STAGING_BUCKET)

job_display_name = "gemma-peft-finetune-job-llm-garage" # Store display name in a variable

job = aiplatform.CustomContainerTrainingJob(
    display_name=job_display_name, # Use the variable here
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
print(f"Submitting training job: {job_display_name}") # Use the variable for pre-submission logging
job.run(
    base_output_dir=f"{NEW_MODEL_OUTPUT_BUCKET}", # Vertex AI specific outputs in llm-garage
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=training_args,
    replica_count=1,
    sync=False
    # service_account="YOUR_VERTEX_AI_CUSTOM_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com" # Optional: if needed
)

job_id = None

def get_logs(job_id):
    client = logging_v2.Client(project="llm-garage")

    # Construct the log filter
    log_filter = f'resource.labels.job_id="{job_id}"'

    # Fetch log entries matching the filter
    entries = client.list_entries(filter_=log_filter)

    print(f"Logs for job ID {job_id}:\n")

    # Iterate through log entries and print them
    for entry in entries:
        print("=" * 80)
        print(f"Timestamp: {entry.timestamp}")
        print(f"Log Name : {entry.log_name}")
        print(f"Severity : {entry.severity}")

        # Print payload depending on type
        if entry.payload_type == "jsonPayload":
            print("Payload (JSON):")
            print(entry.payload)
        elif entry.payload_type == "textPayload":
            print("Payload (Text):")
            print(entry.payload)
        else:
            print("Payload (Proto):")
            print(entry.payload)


while True:
    # Check the job status
    try:
        job_resource_name = job.resource_name
        job_id = job_resource_name.split("/")[-1]
        break
    except Exception as e:
        print(f"Error checking job id: {e}")
        time.sleep(10)

print(f"Job id: {job_id}")

while True:
    get_logs(job_id)
    time.sleep(10)  # Poll every 10 seconds

PROJECT_ID = "llm-garage" 

# time.sleep(10)  # Wait for a few seconds before checking the job status
# # If you still have `job` from run(sync=False), use it directly
# while True:
#     job._sync_gca_resource()  # Refresh the state of the job from Vertex AI
#     state = job.state
#     print(f"Custom job state: {state.name}")

#     if state.name in ("PIPELINE_STATE_RUNNING", "PIPELINE_STATE_SUCCEEDED", "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
#         print(f"Training job completed with state: {state.name}")
#         break

#     time.sleep(10)
