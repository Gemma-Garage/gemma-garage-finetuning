from google.cloud import logging

client = logging.Client(project="llm-garage")
logger = client.logger("gemma-finetune-logs")

logger.log_text("This is a test log from python")

print("Log sent")