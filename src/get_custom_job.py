import threading
import sys
import io
import time
import re
from queue import Queue, Empty
from google.cloud import aiplatform

NEW_STAGING_BUCKET = "gs://llm-garage-vertex-staging"
NEW_DATA_BUCKET = "gs://llm-garage-datasets"
NEW_MODEL_OUTPUT_BUCKET = "gs://llm-garage-models/gemma-peft-vertex-output"

aiplatform.init(
    project="llm-garage",
    location="us-central1",
    staging_bucket=NEW_STAGING_BUCKET,
)

job_display_name = "gemma-peft-finetune-job-llm-garage"

job = aiplatform.CustomContainerTrainingJob(
    display_name=job_display_name,
    container_uri="gcr.io/llm-garage/gemma-finetune:latest",
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/pytorch-gpu.1-13:latest",
)

training_args = [
    f"--dataset={NEW_DATA_BUCKET}/questions.json",
    f"--output_dir={NEW_MODEL_OUTPUT_BUCKET}/model/",
    "--model_name=google/gemma-3-1b-it",
    "--epochs=1",
    "--learning_rate=0.0002",
    "--lora_rank=4",
]

# Queue to store lines printed to stdout
output_queue = Queue()

class QueueWriter(io.TextIOBase):
    def write(self, s):
        # Put every line into queue for monitoring thread
        for line in s.splitlines():
            output_queue.put(line)
        return len(s)

def monitor_output():
    pattern = re.compile(
        r"View backing custom job:\s*(https://console\.cloud\.google\.com/ai/platform/locations/us-central1/training/(\d+)\?project=[^ ]+)"
    )
    print("Monitor thread started: scanning output for custom job URL...")

    while True:
        try:
            line = output_queue.get(timeout=1)  # wait max 1 second for a new line
            print(f"[Monitor] Read line: {line}")  # Debugging line
            # Uncomment to see all lines (optional)
            # print(f"[Monitor] Read line: {line}")

            match = pattern.search(line)
            if match:
                url = match.group(1)
                job_id = match.group(2)
                print(f"[Monitor] Found custom job URL: {url}")
                print(f"[Monitor] Extracted custom job ID: {job_id}")
                break  # exit after finding once

        except Empty:
            # If job.run() has ended, the main thread can set a flag to stop monitor (optional)
            # For now, keep waiting
            continue

    print("Monitor thread exiting.")

# Replace sys.stdout with our queue writer
sys.stdout = QueueWriter()

# Start monitor thread BEFORE running the job synchronously
monitor_thread = threading.Thread(target=monitor_output)
monitor_thread.start()

print(f"Starting training job synchronously: {job_display_name}")
job.run(
    base_output_dir=NEW_MODEL_OUTPUT_BUCKET,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=training_args,
    replica_count=1,
    sync=True,
)

print("Job finished running.")

# Wait for monitor thread to complete if it hasn't already
monitor_thread.join()
print("Monitoring complete.")

# Restore stdout if needed
# sys.stdout = sys.__stdout__
