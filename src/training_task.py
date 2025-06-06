import argparse
import os
# Ensure finetuning.py is in the same directory (src/) or Python path is set correctly
from finetuning import FineTuningEngine
from finetuning_unsloth import UnslothFineTuningEngine 

def training_task(dataset,
                  output_dir,
                  model_name,
                  epochs,
                  learning_rate,
                  lora_rank,
                  request_id,
                  project_id):
    
    print("Starting training task with arguments:")
    print(f"  Dataset: {dataset}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Model Name: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  LoRA Rank: {lora_rank}")
    print(f"  Request ID: {request_id}")
    print(f"  Project ID: {project_id}")

    # Initialize FineTuningEngine
    # Pass request_id and project_id for custom logging
    engine = UnslothFineTuningEngine(
        model_name=model_name,
        request_id=request_id,
        project_id=project_id
    )

    # Modify FineTuningEngine to accept dataset_path in set_lora_fine_tuning
    # and use it directly with load_dataset.
    # Also, ensure perform_fine_tuning saves outputs to args.output_dir.
    # Example (conceptual changes needed in finetuning.py):
    # In finetuning.py, set_lora_fine_tuning might look like:
    #   def set_lora_fine_tuning(self, dataset_path=None, learning_rate=2e-4, ..., output_dir_for_results=None):
    #       self.output_dir_for_results = output_dir_for_results # Store for saving later
    #       if dataset_path:
    #           file_ext = dataset_path.split('.')[-1] if '.' in dataset_path else 'json'
    #           loaded_dataset = load_dataset(file_ext, data_files=dataset_path, split="train")
    #       else: # fallback to default
    #           loaded_dataset = load_dataset("King-Harry/NinjaMasker-PII-Redaction-Dataset", split="train", trust_remote_code=True)
    #       self.dataset = loaded_dataset
    #       # ... rest of the setup ...
    #       training_params.output_dir = self.output_dir_for_results # Ensure HF Trainer also knows output dir

    # In finetuning.py, perform_fine_tuning might look like:
    #   def perform_fine_tuning(self, update_callback=None): # update_callback likely not used
    #       # ...
    #       self.trainer.train()
    #       self.trainer.model.save_pretrained(self.output_dir_for_results) # Use the stored output_dir
    #       self.trainer.tokenizer.save_pretrained(self.output_dir_for_results)


    # print("Setting up LoRA fine-tuning...")
    # engine.set_lora_fine_tuning(
    #     dataset_path=dataset, 
    #     learning_rate=learning_rate,
    #     epochs=epochs,
    #     lora_rank=lora_rank,
    #     #callback_loop=None, # WebSocket not used in Vertex AI
    #     # Pass output_dir to be used by save_pretrained within FineTuningEngine
    #     # This requires modifying FineTuningEngine's set_lora_fine_tuning or __init__
    #     # For now, assuming finetuning.py is adapted to use args.output_dir for saving.
    #     # A common pattern is for the training_params.output_dir to be set to args.output_dir
    #     # and then trainer.save_model() will use that.
    #     output_dir_for_results=output_dir 
    # )
    
    print("Performing fine-tuning...")
    # Ensure perform_fine_tuning in finetuning.py saves to args.output_dir
    engine.train_with_unsloth(
        dataset_path=dataset, 
        learning_rate=learning_rate,
        epochs=epochs,
        lora_rank=lora_rank,
        output_dir_for_results=output_dir
    ) 

    print(f"Training finished. Outputs should be in {output_dir}")

# if __name__ == '__main__':
#     training_task()





# def training_task():
#     parser = argparse.ArgumentParser(description="Vertex AI Fine-tuning Task")
#     parser.add_argument('--dataset', type=str, required=True, help='GCS path to the dataset file or directory')
#     parser.add_argument('--output_dir', type=str, required=True, help='GCS path to save model outputs')
#     parser.add_argument('--model_name', type=str, default='google/gemma-2b', help='Name of the Hugging Face model to finetune')
#     parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
#     parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
#     parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank')
#     parser.add_argument('--request_id', type=str, required=True, help='Unique request ID for this training job')
#     parser.add_argument('--project_id', type=str, required=True, help='Google Cloud Project ID for logging')
#     # Add any other parameters your FineTuningEngine or training process needs

#     args = parser.parse_args()

#     print("Starting training task with arguments:")
#     print(f"  Dataset: {args.dataset}")
#     print(f"  Output Directory: {args.output_dir}")
#     print(f"  Model Name: {args.model_name}")
#     print(f"  Epochs: {args.epochs}")
#     print(f"  Learning Rate: {args.learning_rate}")
#     print(f"  LoRA Rank: {args.lora_rank}")
#     print(f"  Request ID: {args.request_id}")
#     print(f"  Project ID: {args.project_id}")

#     # Initialize FineTuningEngine
#     # Pass request_id and project_id for custom logging
#     engine = FineTuningEngine(
#         model_name=args.model_name,
#         request_id=args.request_id,
#         project_id=args.project_id
#     )

