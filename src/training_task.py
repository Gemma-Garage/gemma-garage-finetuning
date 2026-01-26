"""Gemma Garage Fine-tuning Entry Point - invoked by Cloud Run Jobs"""

import argparse
from finetuning_unsloth import UnslothFineTuningEngine
from rl_finetuning import RLFinetuningEngine


def training_task():
    parser = argparse.ArgumentParser(description="Fine-tuning Task")
    parser.add_argument('--dataset', type=str, required=True, help='GCS path to the dataset file or directory')
    parser.add_argument('--output_dir', type=str, required=True, help='GCS path to save model outputs')
    parser.add_argument('--model_name', type=str, default='google/gemma-2b', help='Name of the Hugging Face model to finetune')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=4, help='LoRA rank')
    parser.add_argument('--request_id', type=str, required=True, help='Unique request ID for this training job')
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud Project ID for logging')
    parser.add_argument('--job_type', type=str, default='supervised', help='Type of training job: supervised or rl_finetuning')
    parser.add_argument('--custom_rubric', type=str, default='', help='Custom rubric for RL reward evaluation')

    args = parser.parse_args()

    dataset = args.dataset
    output_dir = args.output_dir
    model_name = args.model_name
    epochs = args.epochs
    learning_rate = args.learning_rate
    lora_rank = args.lora_rank
    request_id = args.request_id
    project_id = args.project_id
    job_type = args.job_type
    custom_rubric = args.custom_rubric

    print("Starting training task with arguments:")
    print(f"  Dataset: {dataset}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Model Name: {model_name}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  LoRA Rank: {lora_rank}")
    print(f"  Request ID: {request_id}")
    print(f"  Project ID: {project_id}")
    print(f"  Job Type: {job_type}")
    if job_type == "rl_finetuning":
        print(f"  Custom Rubric: {custom_rubric[:100]}..." if len(custom_rubric) > 100 else f"  Custom Rubric: {custom_rubric}")

    # Initialize appropriate engine based on job type
    if job_type == "rl_finetuning":
        # Use RL finetuning engine
        engine = RLFinetuningEngine(
            model_name=model_name,
            request_id=request_id,
            project_id=project_id
        )
    else:
        # Use supervised finetuning engine
        engine = UnslothFineTuningEngine(
            model_name=model_name,
            request_id=request_id,
            project_id=project_id
        )

    print("Performing fine-tuning...")
    # Use appropriate training method based on job type
    if job_type == "rl_finetuning":
        # RL finetuning with custom rubric
        engine.train(
            dataset_path=dataset,
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            lora_rank=lora_rank,
            output_dir_for_results=output_dir,
            custom_rubric=custom_rubric
        )
    else:
        # Supervised finetuning
        engine.train_with_unsloth(
            dataset_path=dataset, 
            learning_rate=learning_rate,
            num_train_epochs=epochs,
            lora_rank=lora_rank,
            output_dir_for_results=output_dir
        ) 

    print(f"Training finished. Outputs should be in {output_dir}")


if __name__ == '__main__':
    training_task()