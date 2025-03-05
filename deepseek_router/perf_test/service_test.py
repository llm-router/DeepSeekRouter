import logging
import random
from typing import Any, List
from datetime import datetime
from openai import OpenAI
from transformers import AutoTokenizer
from pathlib import Path
import json
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from deepseek_router.model.model import ServiceTestResult

services_file_path = Path(__file__).parent / "../../data/services.json"
with services_file_path.open("r") as services_file:
    services = json.load(services_file)

models_file_path = Path(__file__).parent / "../../data/models.json"
with models_file_path.open("r") as models_file:
    models = json.load(models_file)

keys_file_path = Path(__file__).parent / "../../config/keys.json"
with keys_file_path.open("r") as keys_file:
    keys = json.load(keys_file)


class ServiceTest:
    def __init__(self, model_name: str, provider_name: str, api_key: str):
        service = services[model_name][provider_name]
        model_metadata = models[model_name]
        base_url = service["base_url"]
        if provider_name == 'azure':
            self.client = ChatCompletionsClient(endpoint=base_url, credential=AzureKeyCredential(api_key))
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=model_metadata['timeout'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_metadata['tokenizer'])
        self.temperature = model_metadata['temperature']
        self.max_tokens = model_metadata['max_tokens']
        self.provider_name = provider_name
        self.model_name = model_name
        self.model = service["model"]
        self.reasoning_inline = service['reasoning_inline']
        self.reasoning_start = service['reasoning_start'] if self.reasoning_inline else None
        self.reasoning_end = service['reasoning_end'] if self.reasoning_inline else None

    def run_test(self, prompt: str, prompt_type: str, context: List[Any] = None) -> ServiceTestResult:
        if context is None:
            context = list()
        messages = context + [{'role': 'user', 'content': prompt}]
        input_tokens_count = len(self.tokenizer.apply_chat_template(messages, tokenize=True))
        reasoning_content = ''
        final_content = ''
        is_first_token = True
        is_first_result_token = True
        reasoning_started = False
        reasoning_ended = False
        start_time = datetime.now()
        time_to_first_token = None
        time_to_first_result_token = None
        input_tokens_billing = 0
        output_tokens_billing = 0
        cached_tokens_count_billing = 0
        output_tokens_count = 0
        reasoning_tokens_count = 0
        final_tokens_count = 0
        total_time = 0.0
        error = ''

        try:
            if self.provider_name == 'azure':
                response = self.client.complete(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                    max_tokens=self.max_tokens
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                    stream_options={"include_usage": True},
                    max_tokens=self.max_tokens
                )

            for chunk in response:
                current_time = datetime.now()

                # retrieve usage
                if chunk.usage:
                    input_tokens_billing = chunk.usage.prompt_tokens
                    output_tokens_billing = chunk.usage.completion_tokens
                    if getattr(chunk.usage, 'prompt_tokens_details', None) is None:
                        cached_tokens_count_billing = 0
                    else:
                        cached_tokens_count_billing = chunk.usage.prompt_tokens_details.cached_tokens

                # retrieve content
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if self.reasoning_inline:
                    tokens = delta.content
                    if tokens is None:
                        tokens = ''
                    while len(tokens) > 0:
                        if self.reasoning_start in tokens:
                            reasoning_started = True
                            tokens = tokens.replace(self.reasoning_start, "")
                        elif self.reasoning_end in tokens:
                            tokens_arr = tokens.split(self.reasoning_end)
                            reasoning_content += tokens_arr[0] if tokens_arr[0] is not None else ''
                            reasoning_ended = True
                            tokens = tokens_arr[1] if tokens_arr[1] is not None else ''
                        else:
                            if reasoning_started == reasoning_ended:
                                final_content += tokens
                            else:
                                reasoning_content += tokens
                            tokens = ""
                else:
                    reasoning_delta = getattr(delta, "reasoning_content", '')
                    if reasoning_delta is not None and len(reasoning_delta) > 0:
                        reasoning_content += reasoning_delta
                    if delta.content is not None:
                        final_content += delta.content

                if is_first_token and len(reasoning_content) + len(final_content) > 0:
                    time_to_first_token = (current_time - start_time).total_seconds()
                    is_first_token = False
                if is_first_result_token and len(final_content) > 0:
                    time_to_first_result_token = (current_time - start_time).total_seconds()
                    is_first_result_token = False

        except Exception as e:
            error_template = "{0} :\n{1!r}"
            error = error_template.format(type(e).__name__, e.args)
            logging.error("An error occurred", exc_info=True)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        if len(reasoning_content) > 0:
            reasoning_tokens = self.tokenizer(reasoning_content)
            reasoning_tokens_count = len(reasoning_tokens['input_ids'])
        if len(final_content) > 0:
            final_tokens = self.tokenizer(final_content)
            final_tokens_count = len(final_tokens['input_ids'])
        output_tokens_count = reasoning_tokens_count + final_tokens_count

        result = ServiceTestResult(
            provider_name=self.provider_name,
            model_name=self.model_name,
            prompt=json.dumps(messages),
            prompt_type=prompt_type,
            reasoning_content=reasoning_content,
            final_content=final_content,
            success=(len(error) == 0 and output_tokens_billing > 0),
            start_time=start_time,
            end_time=end_time,
            time_to_first_token=time_to_first_token,
            time_to_first_result_token=time_to_first_result_token,
            total_time=total_time,
            input_tokens_count=input_tokens_count,
            output_tokens_count=output_tokens_count,
            reasoning_tokens_count=reasoning_tokens_count,
            cached_tokens_count_billing=cached_tokens_count_billing,
            input_tokens_count_billing=input_tokens_billing,
            output_tokens_count_billing=output_tokens_billing,
            error=error
        )
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate the performance of specified model on the specified dataset.')
    parser.add_argument('--provider', type=str, help='The provider name.', required=True)
    args = parser.parse_args()
    my_provider_name = args.provider
    service_test = ServiceTest('deepseek-r1', my_provider_name, keys[my_provider_name]['default'])
    test_prompt = ('\nEvery morning, Aya does a $9$ kilometer walk, and then finishes at the coffee shop. One day, '
                   'she walks at $s$ kilometers per hour, and the walk takes $4$ hours, including $t$ minutes at the '
                   'coffee shop. Another morning, she walks at $s+2$ kilometers per hour, and the walk takes $2$ hours '
                   'and $24$ minutes, including $t$ minutes at the coffee shop. This morning, if she walks at '
                   '$s+\\frac12$ kilometers per hour, how many minutes will the walk take, including the $t$ minutes '
                   'at the coffee shop?\n\nPlease reason step by step, and put your final answer within \\boxed{}.')
    print(service_test.run_test(prompt=test_prompt, prompt_type='test'))
