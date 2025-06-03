from fastapi import FastAPI, BackgroundTasks, Request
import time
from training_task import training_task

app = FastAPI()

def process_job(param1, param2):
    print(f"Processing with {param1} and {param2}")
    time.sleep(30)  # Simulate long-running task
    print("Job completed")

@app.post("/run-finetune-job")
async def run_job(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    dataset = body.get("dataset")
    output_dir = body.get("output_dir")
    model_name = body.get("model_name")
    epochs = body.get("epochs")
    learning_rate = body.get("learning_rate")
    lora_rank = body.get("lora_rank")
    request_id = body.get("request_id")
    project_id = body.get("project_id")

    background_tasks.add_task(training_task, dataset, 
                              output_dir,
                              model_name,
                              epochs,
                              learning_rate,
                              lora_rank,
                              request_id,
                              project_id)
    return {"status": "Job started"}
