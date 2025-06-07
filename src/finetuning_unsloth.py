
import os


#We need this to track training progress and log it to Google Cloud Logging
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from google.cloud import logging as cloud_logging
from transformers import (
    TrainingArguments,
    TrainerCallback,
)
import math
from datetime import datetime, timezone

# Gemma-specific configurations (can be adjusted)
MAX_SEQ_LENGTH = 2048 # Choose any! We auto support RoPE Scaling internally!
DTYPE = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True # Use 4bit quantization to reduce memory usage. Can be False.

# Define the Alpaca prompt template components - REMOVING THESE
# This can be customized
# PROMPT_TEMPLATE_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
#
# ### Instruction:
# {}
#
# ### Input:
# {}
#
# ### Response:
# {}"""
#
# PROMPT_TEMPLATE_WITHOUT_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
#
# ### Instruction:
# {}
#
# ### Response:
# {}"""

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
    

class UnslothFineTuningEngine:

    def __init__(self, model_name: str, request_id: str, project_id: str = "llm-garage"):
        self.datasets = []
        self.model_name = model_name
        self.request_id = request_id # Store request_id
        self.trainer = None
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


    def train_with_unsloth(
        self,
        dataset_path: str,
        output_dir_for_results: str = "outputs",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 10,
        logging_steps: int = 10,
        save_steps: int = 50,
    ):
        """
        Fine-tunes a model using Unsloth with PEFT LoRA.

        Args:
            dataset_path (str): Path to the training dataset (e.g., a JSONL file).
                                The dataset should have a 'text' column or be processable by a formatting function.
            model_name (str): The Hugging Face model identifier.
            output_dir (str): Directory to save the trained model and logs.
            lora_r (int): LoRA rank.
            lora_alpha (int): LoRA alpha.
            lora_dropout (float): LoRA dropout.
            lora_target_modules (list): List of module names to apply LoRA to.
            learning_rate (float): Initial learning rate.
            num_train_epochs (int): Total number of training epochs.
            per_device_train_batch_size (int): Batch size per GPU.
            gradient_accumulation_steps (int): Number of updates steps to accumulate before performing a backward/update pass.
            warmup_steps (int): Number of steps for the warmup phase.
            logging_steps (int): Log every X updates steps.
            save_steps (int): Save checkpoint every X updates steps.
            hf_token (str): Hugging Face API token for private models.
        """
        print(f"Starting Unsloth fine-tuning for model: {self.model_name} with dataset: {dataset_path}")

        # Load dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        # Load model and tokenizer
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = DTYPE,
            load_in_4bit = LOAD_IN_4BIT,
            #token = hf_token, # Pass Hugging Face token if model is private
            # device_map = "auto", # Unsloth handles device mapping
        )
        print("Model and tokenizer loaded.")

        model = FastLanguageModel.get_peft_model(
            model,
            r = lora_rank,
            target_modules = lora_target_modules,
            lora_alpha = lora_alpha,
            lora_dropout = lora_dropout,
            bias = "none",    # Bias type for LoRA. Can be 'none', 'all' or 'lora_only'
            use_gradient_checkpointing = True, # True or "unsloth" for Unsloth version, helps with memory
            random_state = 3407,
            use_rslora = False, # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
        print("PEFT model configured with LoRA.")
        print(f"train_with_unsloth called with num_train_epochs: {num_train_epochs}")
        # 4. Set up TrainingArguments
        training_args = SFTConfig(
            output_dir=output_dir_for_results,
            num_train_epochs=num_train_epochs,
            dataset_text_field="text",  # Ensure this matches the output of your formatting function or your dataset's text column
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            save_steps=save_steps,
            fp16 = not torch.cuda.is_bf16_supported(), # Use FP16 if BF16 is not available
            bf16 = torch.cuda.is_bf16_supported(),
            optim = "adamw_8bit", # Uses 8-bit AdamW optimizer from bitsandbytes
            lr_scheduler_type="linear",
            seed=3407,
            report_to="tensorboard", # or "wandb"
            max_seq_length=MAX_SEQ_LENGTH
        )

        callbacks = [CloudLoggingCallback(self.cloud_logger, self.request_id)]
        # 5. Create SFTTrainer
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
            callbacks=callbacks
        )

        print("SFTTrainer initialized.")
        self.cloud_logger.log_struct({
            "status_message": f"Starting training. Epochs: {num_train_epochs} Model outputs will be saved to: {trainer.args.output_dir}", # Changed
            "request_id": self.request_id,
            "total_steps": trainer.state.max_steps, # Added
            "total_epochs": trainer.args.num_train_epochs, # Added
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, severity="INFO")

        # 6. Start training
        print("Starting training...")
        trainer.train()
        print("Training finished.")

        self.cloud_logger.log_struct({ # Added
            "status_message": "Training loop completed. Saving model adapters and tokenizer...",
            "request_id": self.request_id,
            "output_dir": self.output_dir_for_results,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }, severity="INFO")

        # 7. Save the model
        final_model_path = os.path.join(output_dir_for_results, "final_model")
        print(f"Saving final LoRA model to {final_model_path}")
        model.save_pretrained(final_model_path) # Saves LoRA adapters
        tokenizer.save_pretrained(final_model_path)
    

