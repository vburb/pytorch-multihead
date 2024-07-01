import json
from typing import Dict, List


def ensure_serializable(dictionary):

    for key, value in dictionary.items():
        if isinstance(value, dict):
            # If the value is a dictionary, recursively call ensure_serializable
            ensure_serializable(value)
        else:
            try:
                # Try to serialize the value to JSON
                json.dumps(value)
            except TypeError:
                # If serialization fails, convert the value to a string
                dictionary[key] = str(value)

    return dictionary


def load_jsonl(file_path: str) -> List[Dict]:
    """Loads a jsonl file to a list of dicts."""
    # List to store parsed JSON objects
    data = []

    # Reading data from JSONL file
    with open(file_path, "r") as f:
        for line in f:
            # Parse JSON object from each line
            json_obj = json.loads(line)
            data.append(json_obj)

    return data


def append_full_jsonl(data, filename: str) -> None:
    """Append a full (aka an entire file) json payload to the end of a jsonl file."""

    # Path(save_filepath).mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as f:
        for job in data:
            json_string = json.dumps(job)
            f.write(json_string + "\n")


def append_to_jsonl(data, filename: str) -> None:
    """Append a single json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")
