import json
import asyncio
import os
import random
import yaml
from tqdm import tqdm

from gen_engine_core.generation_functions.engine_wrapper_class import EngineWrapper
from gen_engine_core.control_flow_functions.control_flow_functions import (
    make_id,
    parse_conversation_to_sharegpt_format,
)

with open("./config.yaml", "r") as file:
    obj_conf = yaml.safe_load(file)

OUTPUT_DIR = obj_conf["PATH"]["OUTPUT"]
EXPERIENCES_DIR = obj_conf["PATH"]["EXPERIENCES"]

semaphore = asyncio.Semaphore(obj_conf["SYSTEM"]["CONCURRENCY_LIMIT"])


async def run_task_with_limit(task):
    async with semaphore:
        return await task


def load_experience_files():
    experience_files = []
    for file_name in os.listdir(EXPERIENCES_DIR):
        if file_name.endswith(".yaml"):
            file_path = os.path.join(EXPERIENCES_DIR, file_name)
            with open(file_path, "r") as file:
                experience_data = yaml.safe_load(file)
                generations = experience_data.get("generations", 1)
                description = experience_data.get("description", "")
                dialogue = experience_data.get("dialogue", [])
                experience_files.append((description, dialogue, generations))
    return experience_files


def create_reformat_prompt(generated_conversation):
    reformat_prompt = f"""Please reformat the following conversation: {generated_conversation} Take this exact conversation and reformat it so that it's in one perfect JSON line, without any space or indentations, just like this: {{"conversations": [{{"from": "human", "value": "..."}}, {{"from": "gpt", "value": "..."}}, {{"from": "human", "value": "..."}}, {{"from": "gpt", "value": "..."}}, {{"from": "human", "value": "..."}}, {{"from": "gpt", "value": "..."}}]}}"""
    return reformat_prompt


def is_valid_sharegpt_format(conversation_json):
    print("Debugging is_valid_sharegpt_format:")
    print(json.dumps(conversation_json, indent=2))

    if not isinstance(conversation_json, dict) or "conversations" not in conversation_json:
        print("Conversation JSON is not a dictionary or doesn't contain 'conversations' key.")
        return False

    conversation_sharegpt = conversation_json["conversations"]
    if not isinstance(conversation_sharegpt, list):
        print("'conversations' value is not a list.")
        return False

    expected_from = "human"
    for turn in conversation_sharegpt:
        if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
            print(f"Turn {turn} is not a dictionary or doesn't contain 'from' or 'value' keys.")
            return False
        if turn["from"] != expected_from:
            print(f"Unexpected 'from' value. Expected: {expected_from}, Actual: {turn['from']}")
            return False
        expected_from = "gpt" if expected_from == "human" else "human"

    print("Conversation is in valid ShareGPT format.")
    return True


async def generate_conv(experience, output_file, engine_wrapper, pbar):
    id = make_id()

    # Unpack the experience tuple
    description, dialogue, _ = experience
    
    # Format the dialogue as a string
    dialogue_str = "\n".join([f"{turn['speaker']}: {turn['message']}" for turn in dialogue])

    prompt = f"""Given the following scenario and initial dialogue:

Description: {description}

Initial dialogue:
{dialogue_str}

Generate a highly diverse, different, and unique conversation based on the provided scenario and dialogue. Do not generate multiple choice questions. Make sure to provide a lengthy and detailed explanation for the answer. Carefully lay out each step to solving the problem before answering the question. Keep the entire JSON object on a single line without any line breaks or indentation:
{{"conversations":[{{"from":"human","value":"..."}},{{"from":"gpt","value":"..."}}]}}
Do not rephrase the question, generate an entirely new question and response of the same theme.
Conversation:"""

    # Generate new conversation using the model
    generated_conversation_tuple = await engine_wrapper.submit_chat(
        messages=[{"role": "user", "content": prompt}],
        sampling_params={
            "max_tokens": 2048,
            "temperature": 1.0,
            "top_p": 0.9,
            "stop": None,
            "stream": True,
        },
    )

    generated_conversation = generated_conversation_tuple[0]

    print("\nGenerated Conversation String:")
    print(generated_conversation)
    print("---------")

    max_attempts = 2
    attempt = 1
    while attempt <= max_attempts:
        try:
            if not generated_conversation.strip().endswith("]}"):
                raise ValueError("Generated conversation does not end with ']}''")

            conversation_json = json.loads(generated_conversation.strip())

            print("Conversation JSON before validation:")
            print(json.dumps(conversation_json, indent=2))

            if is_valid_sharegpt_format(conversation_json):
                conversation_sharegpt = conversation_json["conversations"]
                break
            else:
                print("Generated conversation does not have alternating 'from' values.")
                print("Conversation JSON:")
                print(json.dumps(conversation_json, indent=2))
                print("Reformatting...")
                raise ValueError("Invalid ShareGPT format")
        except (json.JSONDecodeError, KeyError, ValueError):
            print(f"Generated conversation does not match the desired format. Reformatting (attempt {attempt})...")

        reformat_prompt = create_reformat_prompt(generated_conversation)
        reformatted_conversation_tuple = await engine_wrapper.submit_chat(
            messages=[{"role": "user", "content": reformat_prompt}],
            sampling_params={
            "max_tokens": 2048,
            "temperature": 1.0,
            "top_p": 0.9,
            "stop": None,
            "stream": True,
            },
        )
        generated_conversation = reformatted_conversation_tuple[0]

        print("\nReformatted Conversation String:")
        print(generated_conversation)
        print("---------")

        attempt += 1

    if attempt > max_attempts:
        print("Failed to generate a correctly formatted conversation after maximum attempts. Skipping this conversation.")
        return

    if conversation_sharegpt[-1]["from"] == "human":
        print("Generated conversation ends with a 'human' entry. Removing the last entry.")
        conversation_sharegpt = conversation_sharegpt[:-1]

    statements_to_exclude = [
        "incapable of experiencing",
        "incapable of human",
        "lacking human",
        "lacking emotion",
        "I do not possess the capacity",
        "programming does not include",
        "not capable of feeling",
        "not equipped with the capability",
        "do not have the capacity",
        "beyond my capabilities"
    ]

    if any(
        statement in turn["value"]
        for turn in conversation_sharegpt
        if turn["from"] == "gpt"
        for statement in statements_to_exclude
    ):
        print("Generated conversation contains excluded statements in 'gpt' entries. Skipping this conversation.")
        return

    print("\n\nGENERATED CONVERSATION")
    print(conversation_sharegpt)
    print("---------")

    with open(output_file, "a") as f:
        f.write(json.dumps({"conversations": conversation_sharegpt}) + "\n")
        pbar.update(1)


async def main():
    print(obj_conf)
    output_file = "generated_conversations.jsonl"

    engine_wrapper = EngineWrapper(
        model=obj_conf["API"]["NVIDIA_MODEL"],
        api_key=obj_conf["API"]["NVIDIA_API_KEY"],
        base_url=obj_conf["API"]["NVIDIA_BASE_URL"],
    )

    experience_files = load_experience_files()
    total_generations = sum(generations for _, _, generations in experience_files)
    print(f"Total conversations to generate: {total_generations}")

    with tqdm(total=total_generations, unit="conversation") as pbar:
        tasks = []
        for description, dialogue, generations in experience_files:
            for _ in range(generations):
                task = asyncio.create_task(
                    run_task_with_limit(
                        generate_conv((description, dialogue, generations), output_file, engine_wrapper, pbar)
                    )
                )
                tasks.append(task)

        await asyncio.gather(*tasks)


asyncio.run(main())