"""RL fine-tuning engine - GRPO training with LLM-as-judge rewards"""

import os
import sys
import json
import re
import time
import random
import math
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from typing import List

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    import torch
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset, Dataset
    from transformers import TrainingArguments, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
except ImportError as e:
    print(f"ML libraries not available: {e}")

try:
    from google.cloud import storage
    from google.cloud import logging as cloud_logging
    import google.generativeai as genai
    from litellm import completion
except ImportError as e:
    print(f"Cloud/API libraries not available: {e}")

# Data validation imports
try:
    from pydantic import BaseModel, Field
except ImportError as e:
    print(f"Warning: Pydantic not available: {e}")
    # Fallback class definitions
    class BaseModel:
        pass
    def Field(**kwargs):
        return None

# Local imports
from finetuning_unsloth import UnslothFineTuningEngine

# Gemma-specific configurations (matching supervised fine-tuning)
MAX_SEQ_LENGTH = 2048
DTYPE = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
LOAD_IN_4BIT = True  # Use 4bit quantization to reduce memory usage



class TrajectoryScore(BaseModel):
    trajectory: str
    score: float = Field(ge=0.0, le=1.0)

class TrajectoryGradingOutput(BaseModel):
    results: List[TrajectoryScore]

class BaseGraderFunction:
    def grade(self, response):
        raise NotImplementedError

def eval_fun(fn):
    class FnWrapper(BaseGraderFunction):
        def grade(self, response):
            return fn(response)
    return FnWrapper()


class MathGrader(BaseGraderFunction):
    def __init__(self):
        self.reasoning_start = "<start_working_out>"
        self.reasoning_end   = "<end_working_out>"
        self.solution_start = "<SOLUTION>"
        self.solution_end = "</SOLUTION>"

        self.match_format = re.compile(
            rf"^[\s]{{0,}}"\
            rf"{self.reasoning_start}.+?{self.reasoning_end}.*?"\
            rf"{self.solution_start}(.+?){self.solution_end}"\
            rf"[\s]{{0,}}$",
            flags = re.MULTILINE | re.DOTALL
            )
        
        self.match_numbers = re.compile(
            rf"{self.solution_start}.*?([\d\.]{{1,}})",
            flags = re.MULTILINE | re.DOTALL
        )

    def match_format_exactly(self, completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Match if format is seen exactly!
            if self.match_format.search(response) is not None: score += 3.0
            scores.append(score)
        return scores
    
    def match_format_approximately(self, completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Count how many keywords are seen - we penalize if too many!
            # If we see 1, then plus some points!
            score += 0.5 if response.count(self.reasoning_start) == 1 else -0.5
            score += 0.5 if response.count(self.reasoning_end)   == 1 else -0.5
            score += 0.5 if response.count(self.solution_start)  == 1 else -0.5
            score += 0.5 if response.count(self.solution_end)    == 1 else -0.5
            scores.append(score)
        return scores
    
    def check_answer(self, prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := self.match_format.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(0)
                continue
            # Correct answer gets 3 points!
            if guess == true_answer:
                score += 3.0
            # Match if spaces are seen
            elif guess.strip() == true_answer.strip():
                score += 1.5
            else:
                # We also reward it if the answer is close via ratios!
                # Ie if the answer is within some range, reward it!
                try:
                    ratio = float(guess) / float(true_answer)
                    if   ratio >= 0.9 and ratio <= 1.1: score += 0.5
                    elif ratio >= 0.8 and ratio <= 1.2: score += 0.25
                    else: score -= 1.0 # Penalize wrong answers
                except:
                    score -= 0.5 # Penalize
            scores.append(score)
        return scores
    
    def grade(self, prompts, completions, answer):
        total_grade = (
            self.match_format_approximately(completions) +
            self.match_format_exactly(completions) +
            self.check_answer(prompts, completions, answer)
        )
        return total_grade

    def check_numbers(self, prompts, completions, answer, **kwargs):
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := self.match_numbers.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0)
                continue
            # Convert to numbers
            try:
                true_answer = float(true_answer.strip())
                guess       = float(guess.strip())
                scores.append(1.5 if guess == true_answer else 0.0)
            except:
                scores.append(0)
                continue
        return scores


class Grader:
    def __init__(self, 
                 grader_funs: list[Callable]):
        self.grader_funs = grader_funs

    def grade(self, input):
        grades = [grader_fun(input) for grader_fun in self.grader_funs]
        return grades.sum()
    
def upload_to_gcs(local_dir, gcs_path):
    bucket_name, *blob_path = gcs_path.replace("gs://", "").split("/", 1)
    blob_path_prefix = blob_path[0] if blob_path else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    for root, _, files in os.walk(local_dir):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, local_dir)
            blob = bucket.blob(os.path.join(blob_path_prefix, relative_path))
            blob.upload_from_filename(full_path)
            print(f"Uploaded {full_path} to gs://{bucket_name}/{blob.name}")


class CloudLoggingCallback(TrainerCallback):
    def __init__(self, cloud_logger, request_id: str): # Add request_id for context if needed
        self.cloud_logger = cloud_logger
        self.request_id = request_id
        self.last_reward = None  # Track the last reward for inclusion in training logs

    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"CloudLoggingCallback: on_log called. request_id: {self.request_id}, logs: {logs}")
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
            }

            # Process logs, converting NaN to "NaN" string
            processed_logs = {}
            for k, v in logs.items():
                if isinstance(v, float) and math.isnan(v):
                    processed_logs[k] = "NaN"
                else:
                    processed_logs[k] = v
            
            # Update payload with processed logs, giving priority to specific keys if needed
            log_payload.update(processed_logs)
            
            # Ensure specific keys like 'loss' and 'learning_rate' are present, even if they were NaN
            if "loss" not in log_payload and "loss" in logs: # If it was NaN and removed
                 log_payload["loss"] = "NaN"
            if "learning_rate" not in log_payload and "learning_rate" in logs: # If it was NaN and removed
                 log_payload["learning_rate"] = "NaN"

            # Add reward field for RL training - include the last known reward or None
            log_payload["reward"] = self.last_reward

            # Remove None values from payload for cleaner logs (NaN strings will be kept)
            log_payload = {k: v for k, v in log_payload.items() if v is not None}

            print(f"CloudLoggingCallback: Logging to cloud: {log_payload}")
            try:
                self.cloud_logger.log_struct(log_payload, severity="INFO")
            except Exception as e:
                print(f"CloudLoggingCallback: Cloud log failed: {e}")
        return control

    def update_reward(self, reward):
        """Update the last known reward value for inclusion in training logs."""
        self.last_reward = reward


class RLFinetuningEngine(UnslothFineTuningEngine):
    """
    Reinforcement Learning Fine-tuning Engine using GRPO (Group Relative Policy Optimization).
    Inherits from UnslothFineTuningEngine and adds RL-specific functionality.
    """
    
    def __init__(self, model_name: str, request_id: str, project_id: str = "llm-garage"):
        super().__init__(model_name, request_id, project_id)
        self.math_grader = MathGrader()
        
        # Log RL engine initialization
        self.cloud_logger.log_struct({
            "status_message": "RL Fine-tuning Engine initialized with GRPO trainer",
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": 0,
            "step_name": "RL Engine Initialization"
        }, severity="INFO")

    def initialize_rl_model(self):
        """Initialize the RL model - placeholder for future enhancements."""
        # This could be used for additional RL-specific model setup
        pass

    def train(self, dataset_path, **kwargs):
        """Main training entry point for RL fine-tuning."""
        return self.train_with_unsloth(dataset_path, **kwargs)


    def define_system_prompt(self):
        reasoning_start = "<start_working_out>"
        reasoning_end   = "<end_working_out>"
        solution_start = "<SOLUTION>"
        solution_end = "</SOLUTION>"

        system_prompt = \
        f"""You are given a problem.
        Think about the problem and provide your working out.
        Place it between {reasoning_start} and {reasoning_end}.
        Then, provide your solution between {solution_start}{solution_end}"""
        return system_prompt
    
    @staticmethod
    def extract_hash_answer(text):
        """Extract answer after #### marker."""
        if "####" not in text: 
            return None
        return text.split("####")[1].strip()
    
    @staticmethod
    def get_gemini_completion(rubric, question, responses, size_group) -> TrajectoryGradingOutput:
        judge_prompt = f"""
            You are a grader evaluating agent responses against a goal rubric.
            Give each trajectory a score between 0 and 1. There are {size_group} responses. You need to output {size_group} scores.

            Rubric:
            {rubric}

            Prompt:

            {question}

            Responses:
            {responses}


            Respond ONLY in the following JSON format:

            {{
            "results": [
                {{"trajectory": "<trajectory 1>", "score": 0.75}},
                ...
            ]
            }}
            """
        
        try:
            response = completion(
                model="gemini/gemini-1.5-flash",  # LiteLLM-style model name for Gemini
                messages=[{"role": "user", "content": judge_prompt}],
                response_model=TrajectoryGradingOutput,  # Enforce Pydantic format
                max_retries=2  # Optional: retry if model returns malformed output
            )
            return response  # this will be a parsed `TrajectoryGradingOutput` instance
        except Exception as e:
            print(f"Validation or generation error: {e}")
            raise

    @staticmethod
    def parse_litellm_json_response(response) -> TrajectoryGradingOutput:
        """Parse LiteLLM response and validate with Pydantic."""
        raw_content = response.choices[0].message.content

        # Remove Markdown code block
        cleaned_json = re.sub(r"^```json|```$", "", raw_content.strip(), flags=re.MULTILINE).strip()

        # Parse JSON
        parsed = json.loads(cleaned_json)

        # Validate with Pydantic
        return TrajectoryGradingOutput(**parsed)
    
    @staticmethod
    def call_with_backoff(func, max_retries=5, *args, **kwargs):
        delay = 0.5  # initial delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):  # customize based on error
                    print(f"Rate limit hit, backing off... attempt {attempt+1}")
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                    delay += random.uniform(0, 0.1)  # jitter
                else:
                    raise e
        raise RuntimeError("Exceeded maximum retries due to rate limiting.")

    def reward_function(self, prompts, completions, **kwargs):
        """Calculate reward scores for RL training using Gemini as a judge."""
        # Prevent model overloading
        time.sleep(0.05)
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]
        size_group = len(responses)
        
        # Use custom rubric if available, otherwise fall back to default
        rubric = getattr(self, 'custom_rubric', None) or "What is correct"
        
        response = self.call_with_backoff(
            self.get_gemini_completion,
            max_retries=5,
            rubric=rubric,
            question=question,
            responses=responses,
            size_group=size_group
        )
        parsed_results = self.parse_litellm_json_response(response)
        
        # Normalize results using z-score
        scores = [result.score for result in parsed_results.results]
        mean = sum(scores) / len(scores)
        std = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
        # For numerical stability
        epsilon = 1e-6
        normalized_scores = scores  # [(x - mean) / (std + epsilon) for x in scores]

        # Calculate average reward for this batch (for logging purposes)
        avg_reward = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0

        # Log reward calculation details
        self.cloud_logger.log_struct({
            "status_message": f"Reward calculation: Question answered with {len(responses)} responses",
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question[:200] + "..." if len(question) > 200 else question,
            "mean_score": mean,
            "std_score": std,
            "scores": scores,
            "avg_reward": avg_reward
        }, severity="INFO")

        # Update the callback with the current reward for inclusion in training logs
        if hasattr(self, 'cloud_logging_callback') and self.cloud_logging_callback:
            self.cloud_logging_callback.update_reward(avg_reward)

        print("============================")
        print(f"QUESTION: {question}")
        print("----------------------------------")
        answer_score_pairs = [(response, normalized_scores[i]) for i, response in enumerate(responses)]
        for answer, score in answer_score_pairs:
            print(f"Answer: {answer} Score: {score}")
        print("============================")

        if len(normalized_scores) != len(completions):
            print(f"Score len: {len(normalized_scores)} completions: {len(completions)}")
            print(normalized_scores)
            print(prompts)
            print(responses)
            raise ValueError("Number of normalized scores does not match the number of completions.")

        return normalized_scores


    def train_with_unsloth(self, dataset_path, 
                           output_dir_for_results="outputs", 
                           lora_rank=16, 
                           lora_alpha=32, 
                           lora_dropout=0.05, 
                           lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                               "gate_proj", "up_proj", "down_proj"], 
                           learning_rate=5e-6, 
                           num_train_epochs=1, 
                           per_device_train_batch_size=1, 
                           gradient_accumulation_steps=1,
                           max_seq_length=2048, 
                           warmup_steps=10, 
                           logging_steps=1, 
                           save_steps=50,
                           max_steps=50,
                           custom_rubric=""):
        """
        Train the model using GRPO (Group Relative Policy Optimization) for RL fine-tuning.
        """

        max_prompt_length = 256
        self.output_dir_for_results = output_dir_for_results
        
        # Store custom rubric for use in reward function
        self.custom_rubric = custom_rubric

        print(f"Starting RL fine-tuning (GRPO) for model: {self.model_name} with dataset: {dataset_path}")
        if custom_rubric:
            print(f"Using custom rubric: {custom_rubric[:100]}..." if len(custom_rubric) > 100 else f"Using custom rubric: {custom_rubric}")

        # Log: Job received
        self.cloud_logger.log_struct({
            "status_message": "RL fine-tuning job received and instantiating engine...",
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": 1,
            "step_name": "RL Job Instantiated"
        }, severity="INFO")

        # Log: Dataset loading started
        self.cloud_logger.log_struct({
            "status_message": "Loading dataset from GCS for RL training...",
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": 2,
            "step_name": "RL Dataset Loading"
        }, severity="INFO")

        # Download from GCS if path starts with gs://
        if dataset_path.startswith("gs://"):
            local_dataset_path = self.download_from_gcs(dataset_path)
        else:
            local_dataset_path = dataset_path

        # Log: Model loading started
        self.cloud_logger.log_struct({
            "status_message": "Loading model and tokenizer for RL training...",
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": 3,
            "step_name": "RL Model Loading"
        }, severity="INFO")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
        )

        tokenizer = get_chat_template(
            tokenizer,
            chat_template="gemma-3")

        print("Model and tokenizer loaded for RL training.")
        print(f"Dataset path: {dataset_path}")
        print(f"Local dataset path: {local_dataset_path}")

        # Log: Dataset formatting
        self.cloud_logger.log_struct({
            "status_message": "Formatting dataset for RL training...",
            "request_id": self.request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": 4,
            "step_name": "RL Dataset Formatting"
        }, severity="INFO")

        # Open the dataset file as JSON to determine its structure
        with open(local_dataset_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as e:
                raise ValueError(f"Failed to load dataset as JSON: {e}")
        
        # If it's a dict with 'qa_pairs' or a list of QA dicts, format and save to temp file
        if isinstance(data, dict) and 'qa_pairs' in data:
            formatted_list = format_for_gemma3_chat(data, tokenizer=tokenizer)
            dataset = Dataset.from_list(formatted_list)
        else:
            dataset = load_dataset("json", data_files=local_dataset_path, split="train")
        
        print("Dataset check for RL training:")
        print(dataset)

        # Configure model for PEFT (Parameter Efficient Fine-Tuning)
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",    # Bias type for LoRA. Can be 'none', 'all' or 'lora_only'
            use_gradient_checkpointing=True, # True or "unsloth" for Unsloth version, helps with memory
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        print("PEFT model configured with LoRA for RL training.")

        # Set up GRPO Configuration for RL training
        training_args = GRPOConfig(
            learning_rate=learning_rate,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_torch_fused",
            logging_steps=logging_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=4,  # Number of generations per prompt for RL
            max_prompt_length=max_prompt_length,
            max_completion_length=max_seq_length - max_prompt_length,
            max_steps=max_steps,
            save_steps=save_steps,
            max_grad_norm=0.1,
            report_to="none",  # Can use Weights & Biases
            output_dir=output_dir_for_results,
        )

        # Enhanced cloud logging callback for RL training
        cloud_logging_callback = CloudLoggingCallback(self.cloud_logger, self.request_id)
        callbacks = [cloud_logging_callback]
        
        # Store callback reference for reward function to update
        self.cloud_logging_callback = cloud_logging_callback
        
        # Create GRPO Trainer for RL fine-tuning
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                self.reward_function,  # Use our custom reward function
            ],
            args=training_args,
            train_dataset=dataset,
            callbacks=callbacks
        )
        
        print("GRPOTrainer initialized for RL fine-tuning.")
        
        # Log comprehensive training start information
        self.cloud_logger.log_struct({
            "status_message": f"Starting RL training with GRPO. Max steps: {max_steps}, Model outputs will be saved to: {trainer.args.output_dir}",
            "request_id": self.request_id,
            "total_steps": trainer.state.max_steps if trainer.state else max_steps,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "num_generations": training_args.num_generations,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": 5,
            "step_name": "RL Training Start"
        }, severity="INFO")

        # Start RL training
        print("Starting RL training with reward-based optimization...")
        trainer.train()
        print("RL training finished.")

        # Log training completion
        self.cloud_logger.log_struct({
            "status_message": "RL training loop completed. Saving model adapters and tokenizer...",
            "request_id": self.request_id,
            "output_dir": self.output_dir_for_results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": 6,
            "step_name": "RL Training Complete"
        }, severity="INFO")

        # Save the RL fine-tuned model
        final_model_path = os.path.join(output_dir_for_results, "final_rl_model")
        print(f"Saving final RL LoRA model to {final_model_path}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            tokenizer.save_pretrained(tmp_dir)
            print(f"Saved RL model locally to: {tmp_dir}")

            # Upload to GCS
            upload_to_gcs(tmp_dir, final_model_path)
                
        print("RL fine-tuned model saved successfully!")
        
        return final_model_path

def format_for_gemma3_chat(data, tokenizer=None, system_prompt="You are a helpful assistant."):
    """
    Converts a dataset to Gemma 3 chat format using the tokenizer's chat template if available.
    - If data is a list of dicts with only a 'text' key, returns as-is (text-only dataset).
    - If data is a dict with 'summary' and 'qa_pairs', applies chat formatting to each QA pair using the tokenizer's chat template (as in the notebook), and outputs a list of dicts with a 'text' key.
    """
    # If data is a list of dicts with only 'text', return as-is (text-only dataset)
    if isinstance(data, list) and all(isinstance(x, dict) and 'text' in x and len(x) == 1 for x in data):
        return data
    # If data is a dict with 'summary' and 'qa_pairs', format the qa_pairs using the tokenizer's chat template
    if isinstance(data, dict) and 'qa_pairs' in data:
        qa_pairs = data['qa_pairs']
        formatted = []
        for item in qa_pairs:
            if 'question' in item and 'answer' in item:
                if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
                    conversation = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": item["question"]},
                        {"role": "assistant", "content": item["answer"]}
                    ]
                    text = tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    if text.startswith("<bos>"):
                        text = text[len("<bos>"):]
                    formatted.append({"text": text})
                else:
                    raise ValueError("Tokenizer with apply_chat_template required for Gemma 3 chat formatting.")
        return formatted
    # Otherwise, fallback: if data is a list of dicts with 'question' and 'answer', treat as QA pairs
    if isinstance(data, list) and all(isinstance(x, dict) and 'question' in x and 'answer' in x for x in data):
        formatted = []
        for item in data:
            if tokenizer is not None and hasattr(tokenizer, 'apply_chat_template'):
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item["question"]},
                    {"role": "assistant", "content": item["answer"]}
                ]
                text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False
                )
                if text.startswith("<bos>"):
                    text = text[len("<bos>"):]
                formatted.append({"text": text})
            else:
                raise ValueError("Tokenizer with apply_chat_template required for Gemma 3 chat formatting.")
        return formatted
    # If data is not in a supported format, raise an error
    raise ValueError("Input data must be a list of text dicts, a dict with 'qa_pairs', or a list of QA dicts.")


# Example usage and main entry point
if __name__ == "__main__":
    """
    Example usage of the RLFinetuningEngine for reinforcement learning fine-tuning.
    
    This demonstrates how to:
    1. Initialize the RL fine-tuning engine
    2. Train a model using GRPO with reward-based optimization
    3. Monitor training progress through cloud logging
    """
    
    # Example configuration
    model_name = "unsloth/gemma-2-2b-it-bnb-4bit"
    request_id = "rl-training-example-001"
    project_id = "llm-garage"
    dataset_path = "gs://your-bucket/path/to/dataset.json"
    
    try:
        # Initialize the RL fine-tuning engine
        print("Initializing RL Fine-tuning Engine...")
        rl_engine = RLFinetuningEngine(
            model_name=model_name,
            request_id=request_id,
            project_id=project_id
        )
        
        # Configure training parameters
        training_config = {
            "output_dir_for_results": "gs://your-bucket/rl-models/",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 5e-6,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 50,
            "logging_steps": 1,
            "save_steps": 25
        }
        
        # Start RL training
        print("Starting RL training with reward-based optimization...")
        final_model_path = rl_engine.train(dataset_path, **training_config)
        
        print(f"RL training completed successfully!")
        print(f"Final model saved to: {final_model_path}")
        
    except Exception as e:
        print(f"RL training failed: {e}")
        raise