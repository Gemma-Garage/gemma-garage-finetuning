# Gemma Garage Fine-tuning

Training engine for fine-tuning Gemma models using LoRA, optimized with Unsloth for reduced memory usage.

## Overview

This service handles both supervised fine-tuning (SFT) and reinforcement learning (GRPO) training jobs. It runs on Cloud Run Jobs with GPU support and integrates with Google Cloud Logging for real-time progress tracking.

## Training Modes

### Supervised Fine-tuning (SFT)

Uses SFTTrainer from TRL with Unsloth optimizations:
- 2x faster training
- 60% less VRAM usage
- 4-bit quantization support

### Reinforcement Learning (GRPO)

Uses Group Relative Policy Optimization with Gemini as a reward model:
- Generates multiple completions per prompt
- Scores responses against a custom rubric
- Updates model based on relative rewards

## Architecture

```
Cloud Run Job (GPU)
      |
      +--> training_task.py (entry point)
             |
             +--> job_type: supervised
             |       +--> UnslothFineTuningEngine
             |       +--> SFTTrainer
             |
             +--> job_type: rl_finetuning
                     +--> RLFinetuningEngine
                     +--> GRPOTrainer
                     +--> Gemini (reward scoring)
```

## Key Components

| File | Purpose |
|------|---------|
| `training_task.py` | Entry point, argument parsing, job routing |
| `finetuning_unsloth.py` | Supervised training with Unsloth |
| `rl_finetuning.py` | RL training with GRPO |
| `finetuning.py` | Legacy training engine (without Unsloth) |

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | - | Base model from HuggingFace |
| `--dataset` | - | GCS path to training data |
| `--epochs` | 1 | Number of training epochs |
| `--learning_rate` | 2e-4 | Learning rate |
| `--lora_rank` | 16 | LoRA adapter rank |
| `--job_type` | supervised | `supervised` or `rl_finetuning` |

## LoRA Configuration

```python
r = 16              # Rank
lora_alpha = 32     # Scaling factor
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]
```

## Dataset Format

**QA Pairs:**
```json
{
  "qa_pairs": [
    {"question": "...", "answer": "..."}
  ]
}
```

**Text Only:**
```json
[
  {"text": "Training example 1"},
  {"text": "Training example 2"}
]
```

## Cloud Integration

- **GCS**: Dataset input, model output
- **Cloud Logging**: Real-time loss metrics, progress tracking
- **HuggingFace Hub**: Model loading, authentication

## Local Development

```bash
pip install -r requirements.txt

python src/training_task.py \
  --dataset gs://bucket/data.json \
  --output_dir gs://bucket/output \
  --model_name google/gemma-2b \
  --epochs 1 \
  --request_id test-123
```

## Deployment

Deployed as a Cloud Run Job via GitHub Actions. Jobs are triggered by the backend service.

## Testing

```bash
pip install -r requirements-test.txt
pytest tests/ -v
```

## Hardware Requirements

- GPU: NVIDIA T4 (16GB) or L4
- RAM: 32GB recommended
- The service uses 4-bit quantization to fit larger models in memory
