import asyncio
import uuid
from openai import OpenAI

def make_id():
    return str(uuid.uuid4())


class EngineWrapper:
    def __init__(
        self,
        model,
        api_key=None,
        base_url=None,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    async def submit_chat(
        self, messages, sampling_params
    ):  # Submit request and wait for it to stream back fully
        # Print the prompt and sampling parameters
        print("\nPrompt:")
        print(messages)
        print("\nSampling Parameters:")
        print(sampling_params)

        if "temperature" not in sampling_params:
            sampling_params["temperature"] = 1
        if "top_p" not in sampling_params:
            sampling_params["top_p"] = 1
        if "max_tokens" not in sampling_params:
            sampling_params["max_tokens"] = 2048
        if "stop" not in sampling_params:
            sampling_params["stop"] = []

        completion = ""
        timed_out = False
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=sampling_params["temperature"],
            top_p=sampling_params["top_p"],
            max_tokens=sampling_params["max_tokens"],
            stream=True,
        )
        for chunk in stream:
            try:
                if chunk.choices[0].delta.content is not None:
                    completion += chunk.choices[0].delta.content
            except:
                print("THIS RESPONSE TIMED OUT PARTWAY THROUGH GENERATION!")
                timed_out = True
        return completion, timed_out