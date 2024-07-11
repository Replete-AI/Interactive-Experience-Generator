import os
import asyncio
import re
import uuid
import yaml
from gen_engine_core.generation_functions.engine_wrapper_class import EngineWrapper
from gen_engine_core.generation_functions.generation_step_class import GenerationStep

with open("./config.yaml", "r") as file:
    obj_conf = yaml.safe_load(file)

OUTPUT_FOLDER = obj_conf["PATH"]["OUTPUT"]
DEFAULT_EXPERIENCE_PATH = obj_conf["PATH"]["DEFAULT_EXPERIENCES"]
CONCURRENCY_LIMIT = obj_conf["SYSTEM"]["CONCURRENCY_LIMIT"]

def make_id():
    return str(uuid.uuid4())

def write_output_to_file(output, directory, uuid):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, f"{uuid}.txt")

    with open(file_path, "w") as file:
        file.write(output)

    print(f"Output written to {file_path}")

def parse_conversation_to_sharegpt_format(conversation):
    if isinstance(conversation, dict):
        conversation_data = conversation
    else:
        conversation_data = yaml.safe_load(conversation)
    
    if isinstance(conversation_data, str):
        # If conversation_data is a string, assume it's in the ShareGPT format
        sharegpt_conversation = []
        lines = conversation_data.split("\n")
        current_speaker = None
        current_message = ""
        for line in lines:
            if line.startswith("Human: ") or line.startswith("AI: "):
                if current_speaker is not None:
                    sharegpt_conversation.append({
                        "from": current_speaker,
                        "value": current_message.strip()
                    })
                current_speaker = line.split(": ")[0]
                current_message = line.split(": ")[1]
            else:
                current_message += "\n" + line
        if current_speaker is not None:
            sharegpt_conversation.append({
                "from": current_speaker,
                "value": current_message.strip()
            })
    else:
        # If conversation_data is a dictionary, assume it's in the experience YAML format
        sharegpt_conversation = [
            {
                "from": "human" if message["speaker"].lower() == "human" else "gpt",
                "value": message["message"],
            }
            for message in conversation_data
        ]
    
    return sharegpt_conversation