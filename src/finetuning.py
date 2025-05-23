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
from google.cloud import logging as cloud_logging


import asyncio
import time
from transformers import TrainerCallback

#path
WEIGHTS_PATH = './weights/weights.pth'

class CloudLoggingCallback(TrainerCallback):
    def __init__(self, cloud_logger):
        self.cloud_logger = cloud_logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # Format logs as a string, or just convert dict to string
            log_msg = str(logs)
            try:
                self.cloud_logger.log_text(log_msg)
            except Exception as e:
                print(f"Failed to log to cloud: {e}")
        return control

class WebSocketCallback(TrainerCallback):
    def __init__(self, websocket, loop):
        self.websocket = websocket
        self.loop = loop  # Main event loop
        self.last_update = time.time()
        asyncio.create_task(self._check_for_updates())
        self.cloud_logger_client = cloud_logging.Client()
        self.cloud_logger = self.cloud_logger_client.logger("gemma-finetune-logs")

    async def _check_for_updates(self):
        while True:
            await asyncio.sleep(1)  # check every second
            now = time.time()
            if now - self.last_update > 5:  # if no update for 5 seconds
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send_json({"status": "waiting for updates"}), self.loop
                    )
                    self.last_update = now
                except Exception as e:
                    print("Error sending waiting update:", e)
                    break

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.last_update = time.time()
            asyncio.run_coroutine_threadsafe(
                self.websocket.send_json(logs), self.loop
            )
        return control


class FineTuningEngine:

    def __init__(self, model_name, websocket):
        self.datasets = []
        self.model_name = model_name
        self.trainer = None
        self.websocket = websocket
        self.model = self.create_model(self.model_name)
        self.weights_path = WEIGHTS_PATH
        self.output_dir_for_results = None  # Initialize output directory
        self.tokenizer = None  # To store the tokenizer

    def set_websocket(self, websocket):
        self.websocket = websocket

    def load_new_dataset(self, dataset_name:str, file_extension:str='json'):
        path_to_dataset = f'./uploads/{dataset_name}'
        dataset = load_dataset(file_extension, data_files=path_to_dataset, split="train")
        self.datasets.append(dataset)
        return dataset

    def set_lora_fine_tuning(self, 
                             dataset_path=None, 
                             learning_rate=2e-4, 
                             epochs=1, 
                             lora_rank=4,
                             output_dir_for_results=None,  # This will be the GCS path
                             callback_loop=None):
        # if dataset is None:
        #     ccdv_dataset = "King-Harry/NinjaMasker-PII-Redaction-Dataset"
        #     dataset = load_dataset(ccdv_dataset, split="train", trust_remote_code=True)
        #     self.dataset = dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        self.output_dir_for_results = output_dir_for_results # Store the GCS path

        peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        r=lora_rank,
        bias="none",
        task_type="CAUSAL_LM",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True) # Assign to self.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        training_params = TrainingArguments(
        output_dir=self.output_dir_for_results, # Use the GCS path here
        num_train_epochs=epochs,
        per_device_train_batch_size=1, # Consider increasing if your GPU memory allows
        gradient_accumulation_steps=1,
        optim="adamw_torch",
        save_strategy="steps", # Explicitly set save strategy
        save_steps=25, # How often to save checkpoints
        logging_strategy="steps", # Explicitly set logging strategy
        logging_steps=1,          # Log metrics every step
        logging_first_step=True,  # Log metrics at the very first step
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=False, # Set to True if using mixed precision (requires NVIDIA Apex or PyTorch >= 1.6)
        bf16=False, # Set to True if using bfloat16 (requires Ampere or newer GPUs and PyTorch >= 1.10)
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="tensorboard", # You can also add "wandb" or "mlflow" if you use them
        # per_device_eval_batch_size =8 # Only if you have an eval_dataset
        )

        callbacks = [CloudLoggingCallback(self.cloud_logger)]


        trainer = SFTTrainer(
        model=self.model,
        train_dataset=dataset,
        peft_config=peft_params,
        #tokenizer=self.tokenizer, # Pass the tokenizer
        args=training_params,
        #callbacks=[WebSocketCallback(self.websocket, callback_loop)]  # Add the custom callback here
        callbacks=callbacks,  # Add the custom callback here
        )

        self.trainer = trainer

    def perform_fine_tuning(self, update_callback=None):

        if self.trainer is None:
            raise Exception("Error! You must create trainer before fine tuning")
        
        print(f"Starting training. Model outputs (adapters, tokenizer, checkpoints) will be saved to: {self.trainer.args.output_dir}")
        self.trainer.train()
        # SFTTrainer automatically saves the LoRA adapters and tokenizer to the output_dir
        # specified in TrainingArguments during and after training.

        # The old method of saving the full state dict to a local path:
        # self.trainer.model.merge_and_unload()
        # torch.save(self.trainer.model.state_dict(), self.weights_path)
        
        # If you want to ensure a final save of the model adapters and tokenizer explicitly (usually redundant if save_steps > 0 or training completes):
        if self.output_dir_for_results:
            print(f"Ensuring final model adapters and tokenizer are saved to {self.output_dir_for_results}")
            self.trainer.save_model(self.output_dir_for_results) # Saves PEFT model (adapters)
            if self.tokenizer:
                self.tokenizer.save_pretrained(self.output_dir_for_results)
        
        print(f"Training finished. Outputs should be in {self.output_dir_for_results}")


    def create_model(self, model_name:str="princeton-nlp/Sheared-LLaMA-1.3B"):
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     device_map="cpu"
        # )
        model = Gemma3ForCausalLM.from_pretrained(model_name, 
                                          device_map="cpu")
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        return model
