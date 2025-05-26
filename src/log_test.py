from google.cloud import aiplatform
from google.cloud import logging
import ast
from datetime import datetime



def get_logs(
    project_id="llm-garage",
    log_name="projects/llm-garage/logs/gemma-finetune-logs",
    limit=100,
    credentials_path=None
):
    # Auth setup
    if credentials_path:
        credentials = service_account.Credentials.from_service_account_file(credentials_path)
        client = logging_v2.Client(project=project_id, credentials=credentials)
    else:
        client = logging_v2.Client(project=project_id)

    # Query
    filter_str = f'logName="{log_name}"'
    entries = client.list_entries(
        filter_=filter_str,
        order_by=logging_v2.DESCENDING,
        max_results=limit,
    )

    logs = []
    for entry in entries:
        logs.append({
            "timestamp": entry.timestamp.isoformat() if entry.timestamp else None,
            "textPayload": entry.payload if isinstance(entry.payload, str) else None
        })

    return logs


# 🧠 Function to parse logs and extract loss values
def extract_loss_from_logs(logs):
    loss_values = []

    for log in logs:
        text = log.get("textPayload")
        timestamp = log.get("timestamp")

        if text and "loss" in text:
            try:
                parsed = ast.literal_eval(text)

                if isinstance(parsed, dict) and "loss" in parsed:
                    loss = parsed["loss"]
                    loss_values.append((timestamp, loss))
            except (ValueError, SyntaxError):
                continue

    # Sort by timestamp
    loss_values.sort(key=lambda x: datetime.fromisoformat(x[0].replace("Z", "+00:00")))

    # Return as list of dicts
    return [{"timestamp": t, "loss": l} for t, l in loss_values]


# 🎯 Main function to run both steps
if __name__ == "__main__":

    logs = get_logs(
        project_id="llm-garage",
        log_name="projects/llm-garage/logs/gemma-finetune-logs",
        limit=100,
        credentials_path="path/to/your/service-account.json"  # Or None if using ADC
    )

    parsed_losses = extract_loss_from_logs(logs)

    print("Losses in chronological order:")
    for entry in parsed_losses:
        print(entry)