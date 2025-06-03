from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    TrainerCallback,
    Gemma3ForCausalLM
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
import asyncio
import argparse # For command-line arguments
from datetime import datetime, timezone # For timestamps

from google.cloud import logging as cloud_logging
import math # For math.floor


#path
WEIGHTS_PATH = './weights/weights.pth' # This might be less relevant if all outputs go to GCS output_dir

class CloudLoggingCallback(TrainerCallback):
    def __init__(self, cloud_logger, request_id: str): # Add request_id for context if needed
        self.cloud_logger = cloud_logger
        self.request_id = request_id

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            current_epoch_val = math.floor(state.epoch)  # 0 for 1st epoch, 1 for 2nd...
            total_epochs_val = state.num_train_epochs

            # Base payload with progress and status
            log_payload = {
                "request_id": self.request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status_message": f"Training: Epoch {current_epoch_val + 1}/{total_epochs_val}, Step {state.global_step}/{state.max_steps}",
                "current_step": state.global_step,
                "total_steps": state.max_steps,
                "current_epoch": current_epoch_val, # 0-indexed (completed epochs)
                "total_epochs": total_epochs_val,
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
            }

            # Add any other metrics from the 'logs' dictionary, avoiding overwrite
            if logs: # Ensure logs is not None
                for k, v in logs.items():
                    if k not in log_payload:
                        log_payload[k] = v
            
            # Remove None values from payload for cleaner logs
            log_payload = {k: v for k, v in log_payload.items() if v is not None}

            print(f"CloudLoggingCallback: Logging to cloud: {log_payload}")
            try:
                self.cloud_logger.log_struct(log_payload, severity="INFO")
            except Exception as e:
                print(f"CloudLoggingCallback: Cloud log failed: {e}")
        return control

class FineTuningEngine:

    def __init__(self, model_name: str, request_id: str, project_id: str = "llm-garage"):
        self.datasets = []
        self.model_name = model_name
        self.request_id = request_id # Store request_id
        self.trainer = None
        self.model = self.create_model(self.model_name)
        # self.weights_path = WEIGHTS_PATH # Less relevant, main output is output_dir_for_results
        self.output_dir_for_results = None
        self.tokenizer = None
        
        # Initialize Google Cloud Logging client and logger with custom log name
        self.cloud_logger_client = cloud_logging.Client(project=project_id)
        # self.cloud_logger_client.setup_logging() # setup_logging() is for standard Python logging integration
        
        # Construct the custom log name using request_id
        # This MUST match the convention in gemma-garage-backend/finetuning/vertexai.py
        self.log_name = f"gemma_garage_job_logs_{self.request_id}"
        self.cloud_logger = self.cloud_logger_client.logger(self.log_name)
        
        print(f"FineTuningEngine initialized. Logging to: {self.log_name}")
        self.cloud_logger.log_struct({
            "status_message": "Initializing fine-tuning engine...", # Changed
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, severity="INFO")

    def load_new_dataset(self, dataset_path:str, file_extension:str='json'): # dataset_path is now GCS path
        print(f"Loading dataset from: {dataset_path}")
        # load_dataset can directly handle GCS paths if gcsfs is installed (usually is in Vertex AI env)
        dataset = load_dataset(file_extension, data_files=dataset_path, split="train")
        self.datasets.append(dataset) # This logic might need review if only one dataset is used
        return dataset

    def set_lora_fine_tuning(self, 
                             dataset_path=None, # This will be the GCS path to the data file
                             learning_rate=2e-4, 
                             epochs=1, 
                             lora_rank=4,
                             output_dir_for_results=None,  # This will be the GCS path for model outputs
                             ):
        if not dataset_path or not output_dir_for_results:
            raise ValueError("dataset_path and output_dir_for_results must be provided.")

        self.cloud_logger.log_struct({ # Added
            "status_message": "Configuring LoRA and training arguments...",
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, severity="INFO")

        print(f"Setting LoRA fine-tuning. Dataset: {dataset_path}, Output GCS: {output_dir_for_results}")
        
        self.cloud_logger.log_struct({ # Added
            "status_message": f"Loading dataset from GCS path: {dataset_path}...",
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, severity="INFO")
        
        train_dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        self.cloud_logger.log_struct({ # Added
            "status_message": "Dataset loaded successfully.",
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, severity="INFO")
        
        self.output_dir_for_results = output_dir_for_results

        peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Specify target modules for Gemma
        r=lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.cloud_logger.log_struct({ # Added
            "status_message": "Tokenizer initialized.",
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, severity="INFO")

        training_params = TrainingArguments(
            output_dir=self.output_dir_for_results, # GCS path for model, checkpoints etc.
            num_train_epochs=epochs,
            per_device_train_batch_size=1, 
            gradient_accumulation_steps=1,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=25, # Adjust as needed
            logging_strategy="steps",
            logging_steps=1, # Log metrics every step for fine-grained plotting
            logging_first_step=True,
            learning_rate=learning_rate,
            weight_decay=0.001,
            fp16=False, 
            bf16=False, # Enable if supported and desired
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type="constant", # or "cosine", "linear"
            report_to="tensorboard", # Vertex AI captures TensorBoard logs if output_dir is GCS
            # per_device_eval_batch_size=8 # If doing evaluation
        )

        # Pass the initialized cloud_logger and request_id to the callback
        callbacks = [CloudLoggingCallback(self.cloud_logger, self.request_id)]

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset, 
            peft_config=peft_params,
            args=training_params,
            callbacks=callbacks,
        )
        self.trainer = trainer
        print("SFTTrainer initialized.")
        
        total_steps_val = self.trainer.state.max_steps
        total_epochs_val = training_params.num_train_epochs # Or self.trainer.args.num_train_epochs
        
        self.cloud_logger.log_struct({ # Added
            "status_message": "SFTTrainer initialized. Ready for training.",
            "request_id": self.request_id,
            "total_steps": total_steps_val,
            "total_epochs": total_epochs_val,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, severity="INFO")

    def perform_fine_tuning(self, update_callback=None): # update_callback not used

        if self.trainer is None:
            # This should not happen if set_lora_fine_tuning was called
            print("Error! Trainer not initialized before perform_fine_tuning.")
            self.cloud_logger.log_struct({
                "message": "Error: Trainer not initialized.",
                "request_id": self.request_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, severity="ERROR")
            raise Exception("Error! You must create trainer before fine tuning")
        
        print(f"Starting training. Model outputs will be saved to: {self.trainer.args.output_dir}")
        self.cloud_logger.log_struct({
            "status_message": f"Starting training. Model outputs will be saved to: {self.trainer.args.output_dir}", # Changed
            "request_id": self.request_id,
            "total_steps": self.trainer.state.max_steps, # Added
            "total_epochs": self.trainer.args.num_train_epochs, # Added
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, severity="INFO")
        
        self.trainer.train()

        self.cloud_logger.log_struct({ # Added
            "status_message": "Training loop completed. Saving model adapters and tokenizer...",
            "request_id": self.request_id,
            "output_dir": self.output_dir_for_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, severity="INFO")
        
        if self.output_dir_for_results:
            print(f"Ensuring final model adapters and tokenizer are saved to {self.output_dir_for_results}")
            self.trainer.save_model(self.output_dir_for_results) 
            if self.tokenizer:
                self.tokenizer.save_pretrained(self.output_dir_for_results)
        
        print(f"Training finished. Outputs should be in {self.output_dir_for_results}")
        self.cloud_logger.log_struct({
            "status_message": f"Training finished. Outputs in {self.output_dir_for_results}", # Changed
            "request_id": self.request_id,
            "weights_url": self.output_dir_for_results, 
            "status": "complete", 
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, severity="INFO")


    def create_model(self, model_name:str="google/gemma-2b"): # Default to a common Gemma model
        # Consider adding BitsAndBytesConfig for 4-bit/8-bit quantization if needed
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # quantization_config=bnb_config, # Uncomment if using bnb_config
            device_map={"":0} # Automatically map to GPU 0 if available, or CPU
        )
        model.config.use_cache = False # Important for training
        # model.config.pretraining_tp = 1 # May not be needed for all models or SFTTrainer
        return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Gemma model on Vertex AI.")
    parser.add_argument("--dataset", type=str, required=True, help="GCS path to the training dataset (e.g., gs://bucket/data.json)")
    parser.add_argument("--output_dir", type=str, required=True, help="GCS path for output model artifacts (e.g., gs://bucket/output_model_request_id)")
    parser.add_argument("--model_name", type=str, default="google/gemma-2b", help="Name of the Hugging Face model to fine-tune.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank.")
    parser.add_argument("--request_id", type=str, required=True, help="Unique request ID for this training job.")
    parser.add_argument("--project_id", type=str, default="llm-garage", help="Google Cloud Project ID.")
    # Add any other parameters your TrainingArguments or LoraConfig might need

    args = parser.parse_args()

    print(f"Starting fine-tuning script with arguments: {args}")

    try:
        # Instantiate the engine with model_name and the crucial request_id
        engine = FineTuningEngine(
            model_name=args.model_name,
            request_id=args.request_id,
            project_id=args.project_id
        )

        # Configure LoRA fine-tuning
        # dataset_path is the GCS path to the data file itself
        # output_dir_for_results is the GCS path for the model outputs
        engine.set_lora_fine_tuning(
            dataset_path=args.dataset, # GCS path to data file
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            lora_rank=args.lora_rank,
            output_dir_for_results=args.output_dir # GCS path for model outputs
        )

        # Perform the fine-tuning
        engine.perform_fine_tuning()
        
        print(f"Fine-tuning job for request_id {args.request_id} completed successfully.")

    except Exception as e:
        print(f"Error during fine-tuning for request_id {args.request_id}: {e}")
        # Optionally, log this critical failure to the custom log stream if engine is initialized
        try:
            # Attempt to get a logger instance even if full engine init failed,
            # or use a global logger if defined.
            # This is a best-effort error log.
            error_log_client = cloud_logging.Client(project=args.project_id if args.project_id else "llm-garage")
            error_logger_name = f"gemma_garage_job_logs_{args.request_id}" if args.request_id else "gemma_garage_job_logs_unknown_request"
            error_logger = error_log_client.logger(error_logger_name)
            error_logger.log_struct({
                "message": f"Critical error in fine-tuning script: {str(e)}",
                "request_id": args.request_id if args.request_id else "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_details": str(e)
            }, severity="CRITICAL")
        except Exception as log_e:
            print(f"Additionally, failed to write critical error to cloud log: {log_e}")
        raise # Re-raise the exception to mark the Vertex AI job as failed
