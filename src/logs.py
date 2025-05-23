import json
import ast

# Load the logs
with open('logs.json') as f:
    logs = json.load(f)

# Filter logs that contain 'loss' in textPayload or jsonPayload
loss_logs = []

for entry in logs:
    payload = None
    received_time_stamp = None
    # Check textPayload (stringified JSON sometimes lands here)
    if 'textPayload' in entry:
        try:
            payload = ast.literal_eval(entry['textPayload'])
            received_time_stamp = entry['timestamp']
        except Exception:
            print(f"Failed to parse textPayload as JSON {entry['textPayload']}")
            payload = None

    if payload:
        loss_logs.append((received_time_stamp, payload))

# Display filtered logs
for log in loss_logs:
    print(log)
