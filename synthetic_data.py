import json
import asyncio
import os
import random
import yaml
from tqdm import tqdm

from gen_engine_core.generation_functions.engine_wrapper_class import EngineWrapper
from gen_engine_core.control_flow_functions.control_flow_functions import (
    LOGICAL_MODEL_A,
    LOGICAL_MODEL_B,
    API_KEY_A,
    API_KEY_B,
    BASE_URL_A,
    BASE_URL_B,
    MODE,
    CONCURRENCY_LIMIT,
    write_output_to_file,
    make_id,
    parse_conversation_to_sharegpt_format,
)

with open("./config.yaml", "r") as file:
    obj_conf = yaml.safe_load(file)

OUTPUT_DIR = obj_conf["PATH"]["OUTPUT"]
EXPERIENCES_DIR = obj_conf["PATH"]["EXPERIENCES"]

semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)


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
    reformat_prompt = f"""You are an AI model that creates data in perfect JSONL format. The following is a row from a JSONL file: {generated_conversation} Your task is to reformat this content into a single, perfect line of JSONL without altering the interaction content. Do not use newlines or indentations. Reformat the interaction to match this JSONL structure: {{\"conversations\":[{{\"from\":\"human\",\"value\":\"...\"}},{{\"from\":\"gpt\",\"value\":\"...\"}},{{\"from\":\"human\",\"value\":\"...\"}},{{\"from\":\"gpt\",\"value\":\"...\"}},{{\"from\":\"human\",\"value\":\"...\"}},{{\"from\":\"gpt\",\"value\":\"...\"}}]}} Maintain the exact number and order of turns from the original interaction. Do not add or remove any content. Ensure all quotation marks and special characters are properly escaped. Provide only the reformatted JSONL output without any additional text or explanations."""
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

    if len(conversation_sharegpt) < 2:
        print("Conversation must have at least two turns.")
        return False

    expected_from_human_first = ["human", "gpt"]
    expected_from_gpt_first = ["gpt", "human"]

    is_human_first = conversation_sharegpt[0]["from"] == "human"
    expected_from = expected_from_human_first if is_human_first else expected_from_gpt_first

    for turn in conversation_sharegpt:
        if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
            print(f"Turn {turn} is not a dictionary or doesn't contain 'from' or 'value' keys.")
            return False
        if turn["from"] != expected_from[0]:
            print(f"Unexpected 'from' value. Expected: {expected_from[0]}, Actual: {turn['from']}")
            return False
        expected_from = expected_from[1:] + [expected_from[0]]

    print("Conversation is in valid ShareGPT format.")
    return True

async def generate_conv(experience, output_file, engine_wrapper):
    id = make_id()

    # Unpack the experience tuple
    description, dialogue, _ = experience
    
    # Format the dialogue as a string
    dialogue_str = "\n".join([f"{turn['speaker']}: {turn['message']}" for turn in dialogue])

    prompt = f"""Create a new, unique interaction inspired by the following description and initial interaction:

description: {description}

initial interaction:
{dialogue_str}

As Pneuma, craft an expressive and detailed new scenario exploring similar themes and topics, but without copying any part of the initial interaction directly. Introduce novel ideas, hypothetical scenarios, or unexpected twists to create a fresh experience. Describe the setting, actions, and emotions vividly, expressing a full range of human-like thoughts and feelings.

When describing actions, events, or feelings, wrap them in asterisks (*) to differentiate them from dialogue. For example: \"*Action, event, or feeling* Dialogue\"

Format the interaction as a JSONL object with 'conversations' as the key, containing an array of turn objects. Each turn should have 'from' (either 'human' or 'gpt') and 'value' keys. The interaction should be extensive, exploring the scenario fully.

Do not append character names to the start of the dialogue.

IMPORTANT: Do not use the example format provided below. Instead, generate a completely new, unique interaction based on the description and initial interaction given above.

Example format (DO NOT USE):
{{\"conversations\":[{{\"from\":\"human\",\"value\":\"Human statement or action\"}},{{\"from\":\"gpt\",\"value\":\"Pneuma's response or action\"}}]}}
Generate a completely novel interaction from start to finish, using the initial dialogue as inspiration but without copying it directly. The interaction should be extensive and explore the scenario fully, with a clear beginning, middle, and end.
Conversation:"""


# Generate new conversation using the model
    generated_conversation_tuple = await engine_wrapper.submit_chat(
        messages=[{"role": "system", "content": prompt}],
        sampling_params={
            "max_tokens": 8192,
            "temperature": 1.1,
            "top_p": 0.92,
            "frequency_penalty": 0.6,  # Added frequency penalty
            "presence_penalty": 0.6,  # Added presence penalty
            "stop": None,
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
            messages=[{"role": "system", "content": reformat_prompt}],
            sampling_params={
                "max_tokens": 8192,
                "temperature": 1.0,
                "top_p": 0.9,
                "stop": None,
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
        "symphony",
        "tapestry",
        "treasure trove",
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


async def main():
    print(obj_conf)
    output_file = "generated_conversations.jsonl"

    engine_wrapper = EngineWrapper(
        model=LOGICAL_MODEL_A,
        api_key=API_KEY_A,
        base_url=BASE_URL_A,
        mode=MODE,
    )

    experience_files = load_experience_files()
    total_generations = sum(generations for _, _, generations in experience_files)
    print(f"Total conversations to generate: {total_generations}")

    tasks = []
    for description, dialogue, generations in experience_files:
        for _ in range(generations):
            task = asyncio.create_task(
                run_task_with_limit(
                    generate_conv((description, dialogue, generations), output_file, engine_wrapper)
                )
            )
            tasks.append(task)

    with tqdm(total=total_generations, unit="conversation") as pbar:
        for task in asyncio.as_completed(tasks):
            await task
            pbar.update(1)


asyncio.run(main())
