from google.cloud import aiplatform

# === CONFIGURATION ===
PROJECT_ID = "llm-garage"
REGION = "us-central1"
BUCKET_URI = "gs://llm-garage-models/gemma-peft-vertex-output/gemma-peft-finetune-20250522-020707/"  # No file name, just folder path
MODEL_DISPLAY_NAME = "test-endpoint"
ENDPOINT_DISPLAY_NAME = "new-model-endpoint"
MACHINE_TYPE = "n1-standard-4"
PREBUILT_CONTAINER_URI = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-11:latest"  # adjust if using PyTorch or custom model

# === INIT ===
aiplatform.init(project=PROJECT_ID, location=REGION)

# === STEP 1: Upload Model ===
model = aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,
    artifact_uri=BUCKET_URI,
    serving_container_image_uri=PREBUILT_CONTAINER_URI,
)

model.wait()  # Wait for upload to complete
print(f"Model uploaded. Resource name: {model.resource_name}")

# === STEP 2: Create Endpoint ===
endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)
print(f"Endpoint created: {endpoint.resource_name}")

# === STEP 3: Deploy Model to Endpoint ===
deployed_model = model.deploy(
    endpoint=endpoint,
    deployed_model_display_name="deployed-model-v1",
    machine_type=MACHINE_TYPE,
)

print("✅ Model deployed and ready for predictions!")
print(f"Endpoint resource name: {endpoint.resource_name}")
