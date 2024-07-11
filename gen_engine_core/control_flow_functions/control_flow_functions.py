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
COMPLETION_MODE = obj_conf["SYSTEM"]["COMPLETION_MODE"]
LOGICAL_MODEL_A = obj_conf["API"]["LOGICAL_MODEL_A"]
LOGICAL_MODEL_B = obj_conf["API"]["LOGICAL_MODEL_B"]
API_KEY_A = obj_conf["API"]["API_KEY_A"]
API_KEY_B = obj_conf["API"]["API_KEY_B"]
BASE_URL_A = obj_conf["API"]["BASE_URL_A"]
BASE_URL_B = obj_conf["API"]["BASE_URL_B"]
MODE = obj_conf["SYSTEM"]["MODE"]
CONCURRENCY_LIMIT = obj_conf["SYSTEM"]["CONCURRENCY_LIMIT"]

engine_wrapper = EngineWrapper(
    model=LOGICAL_MODEL_A,
    api_key=API_KEY_A,
    base_url=BASE_URL_A,
    mode=MODE,
)

engine_wrapper_large = EngineWrapper(
    model=LOGICAL_MODEL_B,
    api_key=API_KEY_B,
    base_url=BASE_URL_B,
    mode=MODE,
)


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
            for message in conversation_data["dialogue"]
        ]
    
    return sharegpt_conversation


def parse_convo_messages(convo):
    print("==================================")
    return (convo, False)