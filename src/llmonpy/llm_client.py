#   Copyright © 2024 Thomas Edward Burns
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#   Software.
#
#   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import concurrent
import copy
import json
import os
import threading
import time
import uuid
from queue import Queue, Empty

import anthropic
from ai21 import AI21Client
from ai21.models.chat import SystemMessage, UserMessage
from fireworks.client import Fireworks
from fireworks.client.error import RateLimitError
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from groq import Groq
from mistralai import Mistral
from openai import OpenAI
import google.generativeai as genai
from together import Together

from llmonpy.llmonpy_util import fix_common_json_encoding_errors
from ratellmiter.rate_llmiter import RateLimitedService, BucketRateLimiter, RateLlmiterMonitor, SecondTicketBucketListener, \
    LlmClientRateLimitException, llmiter, SecondTicketBucket
from llmonpy.system_services import add_service_to_stop

PROMPT_RETRIES = 5
BASE_RETRY_DELAY = 30  # seconds
DEFAULT_THREAD_POOL_SIZE = 200
TOKEN_UNIT_FOR_COST = 1000000

LLMONPY_API_PREFIX = "LLMONPY_"

MISTRAL_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_SIZE)
ANTHROPIC_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_SIZE)
OPENAI_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_SIZE)
DEEPSEEK_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_SIZE)
GEMINI_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_SIZE)
FIREWORKS_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_SIZE)
TOMBU_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_SIZE)
AI21_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_SIZE)
GROQ_THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_SIZE)
MISTRAL_RATE_LIMITER = BucketRateLimiter(180, "MISTRAL")
FIREWORKS_RATE_LIMITER = BucketRateLimiter(300, "FIREWORKS")
TOMBU_RATE_LIMITER = BucketRateLimiter(1200, "TOMBU_FIREWORKS")
AI21_RATE_LIMITER = BucketRateLimiter(60,  "AI21")


class LLMonPyNoKeyForApiException(Exception):
    def __init__(self, api_key_name):
        super().__init__("No API key found for model " + api_key_name)
        self.api_key_name = api_key_name


class TenacityRateLimitError(Exception):
    def __init__(self):
        super().__init__("Rate limit exceeded")
        print("Rate limit exceeded")
        self.status_code = 429


def get_api_key(api_key_name, exit_on_error=True):
    key = os.environ.get(LLMONPY_API_PREFIX + api_key_name)
    if key is None:
        key = os.environ.get(api_key_name)
    if key is None and exit_on_error:
        raise LLMonPyNoKeyForApiException(api_key_name)
    return key


def backoff_after_exception(attempt):
    delay_time = (attempt + 1) * BASE_RETRY_DELAY
    time.sleep(delay_time)


PROMPT_STATE_WAITING_FOR_TICKET = 1
PROMPT_STATE_HAVE_TICKET = 2
PROMPT_STATE_DONE = 3


class LLMClientPromptStatus:
    def __init__(self, prompt_id: str, client_name: str):
        self.prompt_id = prompt_id
        self.client_name = client_name
        self.start_time = time.time()
        self.done_time = None
        self.state = PROMPT_STATE_WAITING_FOR_TICKET

    def make_done(self, done_time):
        self.done_time = done_time
        self.state = PROMPT_STATE_DONE


class LLMClientStatus:
    def __init__(self, client_name: str):
        self.client_name = client_name
        self.in_flight_prompt_dict = {}
        self.in_flight_count = 0
        self.waiting_for_ticket = 0
        self.completed_prompt_count = 0
        self.exception_count = 0
        self.rate_limit_count = 0
        self.slowest_prompt = 0

    def calculate_all(self, current_time):
        self.in_flight_count = len(self.in_flight_prompt_dict)
        self.calculate_slowest_prompt(current_time)

    def calculate_slowest_prompt(self, current_time):
        slowest = 0
        for prompt_status in self.in_flight_prompt_dict.values():
            prompt_time = current_time - prompt_status.start_time
            if prompt_time > slowest:
                slowest = prompt_time
        self.slowest_prompt = slowest

    def to_dict(self):
        result = copy.copy(vars(self))
        del result["in_flight_prompt_dict"]
        return result


class LLMClientStatusService (SecondTicketBucketListener):
    _instance = None
    REPORT_INTERVAL = 2

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMClientStatusService, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_directory=None):
        self.client_status_dict = {}
        self.status_lock = threading.Lock()
        self.timer = threading.Timer(interval=LLMClientStatusService.REPORT_INTERVAL, function=self.report_status)
        self.completion_time_list: [float] = []
        self.log_directory = log_directory
        self.running = False

    def start(self):
        self.running = True
        self.timer.start()

    def start_timer(self):
        if self.running:
            self.timer = threading.Timer(interval=LLMClientStatusService.REPORT_INTERVAL, function=self.report_status)
            self.timer.start()

    def stop(self):
        self.running = False
        self.timer.cancel()
        if self.log_directory is not None:
            self.write_completion_times()

    def write_completion_times(self):
        if len(self.completion_time_list) > 0:
            second_offset_list = []
            first_time = self.completion_time_list[0]
            for time_in_seconds in self.completion_time_list:
                second_index = int(time_in_seconds - first_time)
                second_offset_list.append(second_index)
            max_offset = second_offset_list[len(second_offset_list) - 1]
            second_count_dict = {}
            for second in second_offset_list:
                if second not in second_count_dict:
                    second_count_dict[second] = 1
                else:
                    second_count_dict[second] += 1
            second_count_list = []
            for i in range(max_offset):
                count = second_count_dict.get(i, 0)
                second_count_list.append(count)
            csv_string = ",".join([str(x) for x in second_count_list])
            with open(self.log_directory + "/second_request_counts.csv", "w") as file:
                file.write(csv_string)

    def log_second_bucket(self, bucket: SecondTicketBucket):
        new_requests = bucket.get_new_requests()
        issued_ticket_list = bucket.get_issued_tickets()
        rate_exception_list = bucket.get_new_rate_exceptions()
        finished_request_list = bucket.get_finished_requests()
        with self.status_lock:
            for request in new_requests:
                self.unsafe_start_prompt(request.user_request_id, request.model_name)
            for request in issued_ticket_list:
                self.unsafe_got_ticket(request.user_request_id, request.model_name)
            for request in rate_exception_list:
                self.unsafe_rate_limit_exceeded(request.user_request_id, request.model_name)
            for request in finished_request_list:
                self.unsafe_prompt_done(request.user_request_id, request.model_name)
                self.completion_time_list.append(request.finished_second_bucket_id)

    def start_prompt(self, prompt_id, client_name):
        with self.status_lock:
            self.unsafe_start_prompt(prompt_id, client_name)

    def unsafe_start_prompt(self, prompt_id, client_name):
        if client_name not in self.client_status_dict:
            self.client_status_dict[client_name] = LLMClientStatus(client_name)
        client_status = self.client_status_dict[client_name]
        client_status.waiting_for_ticket += 1
        prompt_status = LLMClientPromptStatus(prompt_id, client_name)
        client_status.in_flight_prompt_dict[prompt_id] = prompt_status

    def rate_limit_exceeded(self, prompt_id, client_name):
        with self.status_lock:
            self.unsafe_rate_limit_exceeded(prompt_id, client_name)

    def unsafe_rate_limit_exceeded(self, prompt_id, client_name):
        client_status = self.client_status_dict[client_name]
        client_status.waiting_for_ticket += 1
        client_status.rate_limit_count += 1
        prompt_status = client_status.in_flight_prompt_dict[prompt_id]
        prompt_status.state = PROMPT_STATE_WAITING_FOR_TICKET

    def got_ticket(self, prompt_id, client_name):
        with self.status_lock:
            self.unsafe_got_ticket(prompt_id, client_name)

    def unsafe_got_ticket(self, prompt_id, client_name):
        client_status = self.client_status_dict[client_name]
        client_status.waiting_for_ticket -= 1
        prompt_status = client_status.in_flight_prompt_dict.get(prompt_id, None)
        if prompt_status is not None:
            prompt_status.state = PROMPT_STATE_HAVE_TICKET

    def prompt_done(self, prompt_id, client_name):
        current_time = time.time()
        with self.status_lock:
            self.unsafe_prompt_done(prompt_id, client_name)
            self.completion_time_list.append(current_time)

    def unsafe_prompt_done(self, prompt_id, client_name):
        client_status = self.client_status_dict[client_name]
        client_status.completed_prompt_count += 1
        del client_status.in_flight_prompt_dict[prompt_id]

    def prompt_failed(self, prompt_id, client_name):
        with self.status_lock:
            self.prompt_failed(prompt_id, client_name)

    def unsafe_prompt_failed(self, prompt_id, client_name):
        client_status = self.client_status_dict[client_name]
        client_status.completed_prompt_count += 1
        client_status.exception_count += 1
        del client_status.in_flight_prompt_dict[prompt_id]

    def copy_status(self):
        with self.status_lock:
            result = copy.deepcopy(self.client_status_dict)
        return result

    def get_all_status(self):
        frozen_status_dict = self.copy_status()
        client_status_list = []
        all_status = LLMClientStatus("All")
        current_time = time.time()
        for client_status in frozen_status_dict.values():
            client_status.calculate_all(current_time)
            if client_status.slowest_prompt > all_status.slowest_prompt:
                all_status.slowest_prompt = client_status.slowest_prompt
            client_status_list.append(client_status)
            all_status.in_flight_count += client_status.in_flight_count
            all_status.waiting_for_ticket += client_status.waiting_for_ticket
            all_status.completed_prompt_count += client_status.completed_prompt_count
            all_status.exception_count += client_status.exception_count
            all_status.rate_limit_count += client_status.rate_limit_count
        client_status_list.sort(key=lambda x: x.client_name)
        result = [all_status]
        result.extend(client_status_list)
        return result

    def report_status(self):
        all_status_list = self.get_all_status()
        current_status = all_status_list[0]
        in_flight = current_status.in_flight_count
        waiting = current_status.waiting_for_ticket
        rate_exceptions = current_status.rate_limit_count
        completed = current_status.completed_prompt_count
        slowest = current_status.slowest_prompt
        print(f"\n{current_status.client_name} in_flight:{in_flight} waiting_for_ticket:{waiting} completed:{completed} rate_exceptions:{rate_exceptions} slowest:{slowest:.3f}\n")
        for client_status in all_status_list[1:]:
            in_flight = client_status.in_flight_count
            waiting = client_status.waiting_for_ticket
            rate_exceptions = client_status.rate_limit_count
            completed = client_status.completed_prompt_count
            slowest = client_status.slowest_prompt
            print(f"{client_status.client_name} in_flight:{in_flight} waiting_for_ticket:{waiting} completed:{completed} rate_exceptions:{rate_exceptions} slowest:{slowest:.3f}")
        self.start_timer()

    @staticmethod
    def get_instance():
        return LLMClientStatusService._instance


def llm_client_prompt_status_service() -> LLMClientStatusService:
    result = LLMClientStatusService.get_instance()
    return result


class LlmClientJSONFormatException(Exception):
    def __init__(self, raw_text):
        super().__init__("JSON parsing error " + raw_text)
        self.status_code = 500


class LlmClientResponse:
    def __init__(self, response_text, response_dict=None, input_cost=0.0, output_cost=0.0):
        self.response_text = response_text
        self.response_dict = response_dict
        self.input_cost = input_cost
        self.output_cost = output_cost

    def get_response_cost(self):
        result = self.input_cost + self.output_cost
        return result


"""
  LllClient is a base class for all language model clients.  It handles rate limit exceptions for all models.  The
  model client should handle JSON parsing errors -- they tend to be model specific.
"""


class LlmClient(RateLimitedService):
    all_client_list = []

    def __init__(self, model_name, max_input, rate_limiter, thread_pool=None, price_per_input_token=0.0,
                 price_per_output_token=0.0):
        self.model_name = model_name
        self.max_input = max_input
        self.rate_limiter = rate_limiter
        self.thread_pool = thread_pool
        self.price_per_input_token = price_per_input_token
        self.price_per_output_token = price_per_output_token
        LlmClient.all_client_list.append(self)
        if rate_limiter is not None:
            rate_limiter.set_rate_limited_service(self)

    def get_service_name(self) -> str:
        return self.model_name

    def start(self):
        # this should init API.
        pass

    def get_model_name(self):
        return self.model_name

    def ratellmiter_is_llm_blocked(self):
        result = True
        print("Testing if blocked")
        try:
            response = self.do_prompt("Hello? Respond with 'World'","You are a helpful assistant", False,
                                      temp=0.0, max_output=10)
            result = response.response_text is None
            print("Blocked test response: " + str(result))
        except Exception as e:
            print("Blocked test exception: " + str(e))
            result = True
        return result

    def get_ratellmiter(self, model_name: str = None):
        return self.rate_limiter

    def prompt(self, prompt_id, prompt_text, system_prompt=None, json_output=False, temp=0.0,
               max_output=None) -> LlmClientResponse:
        result = None
        result = self.rate_llmiter_prompt(prompt_text, system_prompt, json_output, temp, max_output,
                                         model_name_for_logging=self.model_name, user_request_id=prompt_id)
        return result

    @llmiter(user_request_id_arg="user_request_id", model_name_arg="model_name_for_logging")
    def rate_llmiter_prompt(self, prompt_text, system_prompt=None, json_output=False, temp=0.0,
               max_output=None, user_request_id=None, model_name_for_logging=None) -> LlmClientResponse:
        result = None
        result = self.do_prompt(prompt_text, system_prompt, json_output, temp, max_output)
        if result is None:
            raise LlmClientRateLimitException()
        return result

    """    
    @retry(wait=wait_exponential(multiplier=1, min=5, max=15))
    def tenacity_prompt(self, prompt_id, prompt_text, system_prompt=None, json_output=False, temp=0.0,
               max_output=None) -> LlmClientResponse:
        result = None
        llm_client_prompt_status_service().start_prompt(prompt_id, self.model_name)
        llm_client_prompt_status_service().got_ticket(prompt_id, self.model_name)
        result = self.do_prompt(prompt_text, system_prompt, json_output, temp, max_output)
        llm_client_prompt_status_service().prompt_done(prompt_id, self.model_name)
        return result
    """

    def wait_for_ticket_after_rate_limit_exceeded(self, prompt_id, ticket):
        #llm_client_prompt_status_service().rate_limit_exceeded(prompt_id, self.model_name)
        ticket = self.rate_limiter.wait_for_ticket_after_rate_limit_exceeded(ticket)
        #llm_client_prompt_status_service().got_ticket(prompt_id, self.model_name)
        return ticket

    def do_prompt(self, prompt_text, system_prompt=None, json_output=False, max_output=None, temp=0.0):
        raise Exception("Not implemented")

    def get_thread_pool(self):
        return self.thread_pool

    def calculate_costs(self, input_tokens, output_tokens):
        input_cost = (input_tokens * self.price_per_input_token) / TOKEN_UNIT_FOR_COST
        output_cost = (output_tokens * self.price_per_output_token) / TOKEN_UNIT_FOR_COST
        return input_cost, output_cost

    @staticmethod
    def get_all_clients():
        return LlmClient.all_client_list


class OpenAIModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thread_pool=None, price_per_input_token=0.0,
                 price_per_output_token=0.0):
        super().__init__(model_name, max_input, rate_limiter, thread_pool, price_per_input_token, price_per_output_token)
        self.client = None

    def start(self):
        key = get_api_key("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text.", json_output=False,
                  temp=0.0, max_output=None):
        system_prompt = system_prompt if system_prompt is not None else "You are an expert at analyzing text."
        result = None
        response_text = None
        response_format = "json_object" if json_output else "text"
        # retries just for json format errors
        for attempt in range(PROMPT_RETRIES):
            completion = self.client.chat.completions.create(
                model=self.model_name,
                response_format={"type": response_format},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=temp,
                timeout=90
            )
            response_text = completion.choices[0].message.content
            response_dict = None
            if json_output:
                try:
                    response_text = fix_common_json_encoding_errors(response_text)
                    response_dict = json.loads(response_text)
                except Exception as e:
                    continue
            input_cost, output_cost = self.calculate_costs(completion.usage.prompt_tokens,
                                                           completion.usage.completion_tokens)
            result = LlmClientResponse(response_text, response_dict, input_cost, output_cost)
        if result is None and json_output:
            raise LlmClientJSONFormatException(response_text)
        return result


class DeepseekModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thread_pool=None, price_per_input_token=0.0,
                 price_per_output_token=0.0):
        super().__init__(model_name, max_input, rate_limiter, thread_pool, price_per_input_token, price_per_output_token)
        self.client = None

    def start(self):
        key = get_api_key("DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=key, base_url="https://api.deepseek.com/")

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text.", json_output=False):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.0
        )
        result = completion.choices[0].message.content
        return result


class AnthropicModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thread_pool=None, price_per_input_token=0.0,
                 price_per_output_token=0.0):
        super().__init__(model_name, max_input, rate_limiter, thread_pool, price_per_input_token, price_per_output_token)
        self.client = None

    def start(self):
        key = get_api_key("ANTHROPIC_API_KEY")
        self.client = anthropic.Client(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text.", json_output=False,
                  temp=0.0, max_output=4096):
        system_prompt = system_prompt if system_prompt is not None else "You are an expert at analyzing text."
        max_output = max_output if max_output is not None else 4096
        prompt_messages = [
            {
                "role": "user",
                "content": prompt_text
            }
        ]
        if json_output:
            prompt_messages.append({"role": "assistant", "content": "{"})
        result = None
        response_text = None
        # retries just for json format errors
        for attempt in range(PROMPT_RETRIES):
            message = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_output,
                temperature=temp,
                system=system_prompt,
                messages=prompt_messages
            )
            response_text = message.content[0].text
            response_dict = None
            if json_output:
                try:
                    response_text = "{ " + response_text
                    response_text = fix_common_json_encoding_errors(response_text)
                    response_dict = json.loads(response_text)
                except Exception as e:
                    continue
            input_cost, output_cost = self.calculate_costs(message.usage.input_tokens, message.usage.output_tokens)
            result = LlmClientResponse(response_text, response_dict, input_cost, output_cost)
        if result is None and json_output:
            raise LlmClientJSONFormatException(response_text)
        return result


class MistralLlmClient(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thread_pool, price_per_input_token=0.0,
                 price_per_output_token=0.0):
        super().__init__(model_name, max_input, rate_limiter, thread_pool, price_per_input_token, price_per_output_token)
        self.client = None

    def start(self):
        key = get_api_key("MISTRAL_API_KEY")
        self.client = Mistral(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text.", json_output=False, temp=0.0,
                  max_output=None):
        system_prompt = system_prompt if system_prompt is not None else "You are an expert at analyzing text."
        response_format = "json_object" if json_output else "text"
        prompt_messages = [
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": prompt_text }
        ]
        result = None
        response_text = None
        # retries just for json format errors
        for attempt in range(PROMPT_RETRIES):
            response = self.client.chat.complete(
                model=self.model_name,
                response_format={"type": response_format},
                max_tokens=max_output,
                temperature=temp,
                messages=prompt_messages
            )
            response_text = response.choices[0].message.content
            response_dict = None
            if json_output:
                try:
                    response_text = fix_common_json_encoding_errors(response_text)
                    response_dict = json.loads(response_text)
                except Exception as e:
                    continue
            input_cost, output_cost = self.calculate_costs(response.usage.prompt_tokens,
                                                           response.usage.completion_tokens)
            result = LlmClientResponse(response_text, response_dict, input_cost, output_cost)
        if result is None and json_output:
            raise LlmClientJSONFormatException(response_text)
        return result


# https://ai.google.dev/gemini-api/docs/get-started/tutorial?authuser=2&lang=python
class GeminiModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thread_pool=None, price_per_input_token=0.0,
                 price_per_output_token=0.0):
        super().__init__(model_name, max_input, rate_limiter, thread_pool, price_per_input_token, price_per_output_token)
        self.client = None

    def start(self):
        key = get_api_key("GEMINI_API_KEY")
        genai.configure(api_key=key)
        self.client = genai.GenerativeModel(self.model_name)
        self.json_client = genai.GenerativeModel(self.model_name,
                                                 generation_config={"response_mime_type": "application/json"})

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text.", json_output=False,
                  temp=0.0, max_output=None):
        prompt_client = self.json_client if json_output else self.client
        full_prompt = str(system_prompt) + "\n\n" + prompt_text
        result = None
        response_text = None
        # retries just for json format errors
        for attempt in range(PROMPT_RETRIES):
            model_response = prompt_client.generate_content(full_prompt,
                                                            safety_settings={
                                                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
                                                            },
                                                            generation_config=genai.GenerationConfig(temperature=temp))
            response_text = model_response.text
            response_dict = None
            if json_output:
                try:
                    response_text = fix_common_json_encoding_errors(response_text)
                    response_dict = json.loads(response_text)
                except Exception as e:
                    continue
            input_cost, output_cost = self.calculate_costs(model_response.usage_metadata.prompt_token_count,
                                                           model_response.usage_metadata.candidates_token_count)
            result = LlmClientResponse(response_text, response_dict, input_cost, output_cost)
            if result is None and json_output:
                raise LlmClientJSONFormatException(response_text)
        return result


class TogetherAIModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thread_pool=None, price_per_input_token=0.0,
                 price_per_output_token=0.0):
        super().__init__(model_name, max_input, rate_limiter, thread_pool, price_per_input_token, price_per_output_token)
        self.client = None

    def start(self):
        key = get_api_key("TOGETHER_API_KEY")
        self.client = Together(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text.", json_output=False,
                  temp=0.0, max_output=None):
        system_prompt = system_prompt if system_prompt is not None else ""
        result = None
        response_text = None
        full_prompt = str(system_prompt) + "\n\n" + prompt_text
        # retries just for json format errors
        for attempt in range(PROMPT_RETRIES):
            completion = self.client.completions.create(
                model=self.model_name,
                prompt=full_prompt,
                temperature=temp
            )
            response_text = completion.choices[0].text
            response_dict = None
            if json_output:
                try:
                    response_text = fix_common_json_encoding_errors(response_text)
                    response_dict = json.loads(response_text)
                except Exception as e:
                    print("JSON parsing error " + response_text)
                    continue
            input_cost, output_cost = self.calculate_costs(completion.usage.prompt_tokens,
                                                           completion.usage.completion_tokens)
            result = LlmClientResponse(response_text, response_dict, input_cost, output_cost)
        if result is None and json_output:
            raise LlmClientJSONFormatException(response_text)
        return result


class FireworksAIModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thread_pool=None, price_per_input_token=0.0,
                 price_per_output_token=0.0, system_role_supported=True):
        super().__init__(model_name, max_input, rate_limiter, thread_pool, price_per_input_token, price_per_output_token)
        self.client = None
        self.system_role_supported = system_role_supported

    def start(self):
        key = get_api_key("FIREWORKS_API_KEY")
        self.client = Fireworks(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text.", json_output=False,
                  temp=0.0, max_output=None):
        system_prompt = system_prompt if system_prompt is not None else "You are an expert at analyzing text."
        result = None
        response_text = None
        response_format = "json_object" if json_output else "text"
        # retries just for json format errors
        for attempt in range(PROMPT_RETRIES):
            if self.system_role_supported:
                completion = self.client.chat.completions.create(

                    model=self.model_name,
                    # response_format={"type": response_format},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=temp
                )
            else:
                full_prompt = str(system_prompt) + "\n\n" + prompt_text
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    response_format={"type": response_format},
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=temp
                )
            response_text = completion.choices[0].message.content
            response_dict = None
            if json_output:
                try:
                    response_text = fix_common_json_encoding_errors(response_text)
                    response_dict = json.loads(response_text)
                except Exception as e:
                    continue
            input_cost, output_cost = self.calculate_costs(completion.usage.prompt_tokens,
                                                           completion.usage.completion_tokens)
            result = LlmClientResponse(response_text, response_dict, input_cost, output_cost)
        if result is None and json_output:
            raise LlmClientJSONFormatException(response_text)
        return result

class AI21Model(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thread_pool=None, price_per_input_token=0.0,
                 price_per_output_token=0.0):
        super().__init__(model_name, max_input, rate_limiter, thread_pool, price_per_input_token, price_per_output_token)
        self.client = None

    def start(self):
        key = get_api_key("AI21_API_KEY")
        self.client = AI21Client(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text.", json_output=False,
                  temp=0.0, max_output=None):
        system_prompt = system_prompt if system_prompt is not None else "You are an expert at analyzing text."
        result = None
        response_text = None
        response_format = "json_object" if json_output else "text"
        # retries just for json format errors
        for attempt in range(PROMPT_RETRIES):
            completion = self.client.chat.completions.create(
                model=self.model_name,
                temperature=temp,
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=prompt_text)
                ],
            )
            result = completion.choices[0].message.content
            response_dict = None
            if json_output:
                try:
                    response_text = fix_common_json_encoding_errors(response_text)
                    response_dict = json.loads(response_text)
                except Exception as e:
                    continue
            input_cost, output_cost = self.calculate_costs(completion.usage.prompt_tokens,
                                                           completion.usage.completion_tokens)
            result = LlmClientResponse(response_text, response_dict, input_cost, output_cost)
        if result is None and json_output:
            raise LlmClientJSONFormatException(response_text)
        return result


class GroqModel(LlmClient):
    def __init__(self, model_name, max_input, rate_limiter, thread_pool=None, price_per_input_token=0.0,
                 price_per_output_token=0.0, system_role_supported=True):
        super().__init__(model_name, max_input, rate_limiter, thread_pool, price_per_input_token, price_per_output_token)
        self.client = None
        self.system_role_supported = system_role_supported

    def start(self):
        key = get_api_key("GROQ_API_KEY")
        self.client = Groq(api_key=key)

    def do_prompt(self, prompt_text, system_prompt="You are an expert at analyzing text.", json_output=False,
                  temp=0.0, max_output=None):
        system_prompt = system_prompt if system_prompt is not None else "You are an expert at analyzing text."
        result = None
        response_text = None
        response_format = "json_object" if json_output else "text"
        # retries just for json format errors
        for attempt in range(PROMPT_RETRIES):
            if self.system_role_supported:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    response_format={"type": response_format},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}
                    ],
                    temperature=temp
                )
            else:
                full_prompt = str(system_prompt) + "\n\n" + prompt_text
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    response_format={"type": response_format},
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=temp
                )
            response_text = completion.choices[0].message.content
            response_dict = None
            if json_output:
                try:
                    response_text = fix_common_json_encoding_errors(response_text)
                    response_dict = json.loads(response_text)
                except Exception as e:
                    continue
            input_cost, output_cost = self.calculate_costs(completion.usage.prompt_tokens,
                                                           completion.usage.completion_tokens)
            result = LlmClientResponse(response_text, response_dict, input_cost, output_cost)
        if result is None and json_output:
            raise LlmClientJSONFormatException(response_text)
        return result


# MIXTRAL tokenizer generates 20% more tokens than openai, so after reduce max_input to 80% of openai
MINISTRAL_3B = MistralLlmClient("ministral-3b-latest", 12000, MISTRAL_RATE_LIMITER, MISTRAL_THREAD_POOL, 0.04, 0.04)
MINISTRAL_8B = MistralLlmClient("ministral-8b-latest", 12000, MISTRAL_RATE_LIMITER, MISTRAL_THREAD_POOL, 0.10, 0.10)
MISTRAL_7B = MistralLlmClient("open-mistral-7b", 12000, MISTRAL_RATE_LIMITER, MISTRAL_THREAD_POOL, 0.25, 0.25)
MISTRAL_NEMO_12B = MistralLlmClient("open-mistral-nemo-2407", 12000, MISTRAL_RATE_LIMITER, MISTRAL_THREAD_POOL, 0.15, 0.15)
MISTRAL_8X22B = MistralLlmClient("open-mixtral-8x22b", 8000, MISTRAL_RATE_LIMITER, MISTRAL_THREAD_POOL, 2.0, 6.0)
MISTRAL_SMALL = MistralLlmClient("mistral-small-2409", 24000, MISTRAL_RATE_LIMITER, MISTRAL_THREAD_POOL, 0.2, 0.6)
MISTRAL_8X7B = MistralLlmClient("open-mixtral-8x7b", 24000, MISTRAL_RATE_LIMITER, MISTRAL_THREAD_POOL, 0.7, 0.7)
MISTRAL_LARGE = MistralLlmClient("mistral-large-2407", 120000, MISTRAL_RATE_LIMITER, MISTRAL_THREAD_POOL, 2.0, 6.0)
GPT3_5 = OpenAIModel('gpt-3.5-turbo-0125', 15000, BucketRateLimiter(10000), OPENAI_THREAD_POOL, 0.5, 1.5)
GPT4 = OpenAIModel('gpt-4-turbo-2024-04-09', 120000, BucketRateLimiter(10000), OPENAI_THREAD_POOL, 10.0, 30.0)
GPT4o = OpenAIModel('gpt-4o-2024-08-06', 120000, BucketRateLimiter(10000), OPENAI_THREAD_POOL, 2.5, 10.0)
GPT4omini = OpenAIModel('gpt-4o-mini', 120000, BucketRateLimiter(10000), OPENAI_THREAD_POOL, 0.15, 0.60)
ANTHROPIC_OPUS = AnthropicModel("claude-3-opus-20240229", 180000, BucketRateLimiter(240), ANTHROPIC_THREAD_POOL,
                                15.0, 75.0)
ANTHROPIC_SONNET = AnthropicModel("claude-3-5-sonnet-20241022", 180000, BucketRateLimiter(480),
                                  ANTHROPIC_THREAD_POOL, 3.0, 15.0)
ANTHROPIC_HAIKU = AnthropicModel("claude-3-haiku-20240307", 180000, BucketRateLimiter(240), ANTHROPIC_THREAD_POOL,
                                 0.25, 1.25)
# DEEPSEEK = DeepseekModel("deepseek-chat", 24000, RateLlmiter(20, MINUTE_TIME_WINDOW), DEEPSEEK_EXECUTOR)
OLD_GEMINI_FLASH = GeminiModel("gemini-1.5-flash", 120000, BucketRateLimiter(240), GEMINI_THREAD_POOL, 0.075, .35)
GEMINI_FLASH_2 = GeminiModel("gemini-2.0-flash-exp", 120000, BucketRateLimiter(240), GEMINI_THREAD_POOL, 0.075, .35)
GEMINI_FLASH = GeminiModel("gemini-1.5-flash-002", 120000, BucketRateLimiter(240), GEMINI_THREAD_POOL, 0.075, .35)
GEMINI_FLASH_8B = GeminiModel("gemini-1.5-flash-8b", 120000, BucketRateLimiter(240), GEMINI_THREAD_POOL, 0.0375, .15)
GEMINI_PRO = GeminiModel("gemini-1.5-pro", 120000, BucketRateLimiter(240), GEMINI_THREAD_POOL, 1.25, 2.50)
FIREWORKS_LLAMA3_2_1B = FireworksAIModel("accounts/fireworks/models/llama-v3p2-1b-instruct", 120000,
                                         FIREWORKS_RATE_LIMITER, FIREWORKS_THREAD_POOL, 0.10, 0.20)
FIREWORKS_LLAMA3_2_3B = FireworksAIModel("accounts/fireworks/models/llama-v3p2-3b-instruct", 120000,
                                         FIREWORKS_RATE_LIMITER, FIREWORKS_THREAD_POOL, 0.10, 0.20)
FIREWORKS_LLAMA3_1_8B = FireworksAIModel("accounts/fireworks/models/llama-v3p1-8b-instruct", 120000,
                                         FIREWORKS_RATE_LIMITER, FIREWORKS_THREAD_POOL, 0.20, 0.20)
FIREWORKS_LLAMA3_1_405B = FireworksAIModel("accounts/fireworks/models/llama-v3p1-405b-instruct", 120000,
                                           FIREWORKS_RATE_LIMITER, FIREWORKS_THREAD_POOL, 3.00, 3.00)
FIREWORKS_LLAMA3_1_70B = FireworksAIModel("accounts/fireworks/models/llama-v3p1-70b-instruct", 120000,
                                          FIREWORKS_RATE_LIMITER, FIREWORKS_THREAD_POOL, 0.90, 0.90)
FIREWORKS_GEMMA2_9B = FireworksAIModel("accounts/fireworks/models/gemma2-9b-it", 7500, FIREWORKS_RATE_LIMITER,
                                       FIREWORKS_THREAD_POOL, 0.20, 0.20, system_role_supported=False)
FIREWORKS_MYTHOMAXL2_13B = FireworksAIModel("accounts/fireworks/models/mythomax-l2-13b", 4000, FIREWORKS_RATE_LIMITER,
                                            FIREWORKS_THREAD_POOL, 0.20, 0.20)
FIREWORKS_QWEN2_72B = FireworksAIModel("accounts/fireworks/models/qwen2p5-72b-instruct", 32000, FIREWORKS_RATE_LIMITER,
                                       FIREWORKS_THREAD_POOL, 0.90, 0.90)
FIREWORKS_DEEPSEEK_V3 = FireworksAIModel("accounts/fireworks/models/deepseek-v3", 32000, FIREWORKS_RATE_LIMITER,
                                       FIREWORKS_THREAD_POOL, 0.90, 0.90)
TOMBU_LLAMA3_1_8B = FireworksAIModel("accounts/fireworks/models/llama-v3p1-8b-instruct#accounts/tombu-8c8576/deployments/ffdd8605", 120000,
                                         TOMBU_RATE_LIMITER, TOMBU_THREAD_POOL, 0.20, 0.20)
TOMBU_DOLPHIN_QWEN2_72B= FireworksAIModel("accounts/fireworks/models/dolphin-2-9-2-qwen2-72b#accounts/tombu-8c8576/deployments/39b81ca3", 120000,
                                         TOMBU_RATE_LIMITER, TOMBU_THREAD_POOL, 0.20, 0.20)
TOMBU_NEMO_12B= FireworksAIModel("accounts/fireworks/models/mistral-nemo-instruct-2407#accounts/tombu-8c8576/deployments/0fdd946e", 120000,
                                         TOMBU_RATE_LIMITER, TOMBU_THREAD_POOL, 0.20, 0.20)
GROQ_LLAMA3_1_70B= GroqModel("llama-3.1-70b-versatile", 120000,
                                         BucketRateLimiter(100), GROQ_THREAD_POOL, 0.70, 0.70)
GROQ_LLAMA3_1_8B= GroqModel("llama-3.1-8b-instant", 120000,
                                         BucketRateLimiter(30), GROQ_THREAD_POOL, 0.20, 0.20)
AI21_JAMBA_1_5_MINI = AI21Model("jamba-1.5-mini", 120000,
                                AI21_RATE_LIMITER, AI21_THREAD_POOL,0.20, 0.40)
AI21_JAMBA_1_5_LARGE = AI21Model("jamba-1.5-large", 120000,
                                AI21_RATE_LIMITER, AI21_THREAD_POOL,2.00, 8.00)
ACTIVE_LLM_CLIENT_DICT = {}


def init_llm_clients(data_directory="data"):
    log_directory = os.path.join(data_directory, "rate_llmiter_logs")
    client_list = LlmClient.get_all_clients()
    clients_with_keys = []
    missing_key_map = {}
    for client in client_list:
        try:
            client.start()
            ACTIVE_LLM_CLIENT_DICT[client.model_name] = client
            clients_with_keys.append(client)
        except LLMonPyNoKeyForApiException as key_exception:
            missing_key_map[key_exception.api_key_name] = key_exception.api_key_name
            continue
    for key in missing_key_map:
        print("No key found for " + key)
    status_service = LLMClientStatusService(log_directory)
    status_service.start()
    add_service_to_stop(status_service)
    RateLlmiterMonitor.get_instance().add_listener(status_service)
    RateLlmiterMonitor.get_instance().start()
    add_service_to_stop(RateLlmiterMonitor.get_instance())
    return clients_with_keys

def stop_llm_clients():
    llm_client_prompt_status_service().stop()


def get_active_llm_clients():
    result = list(ACTIVE_LLM_CLIENT_DICT.values())
    result.sort(key=lambda client: client.model_name)
    return result


def filter_clients_that_didnt_start(client_list):
    result = []
    for client in client_list:
        if client.model_name in ACTIVE_LLM_CLIENT_DICT:
            result.append(client)
        else:
            print("Client " + client.model_name + " did not start")
    return result


def get_llm_client(model_name):
    return ACTIVE_LLM_CLIENT_DICT[model_name]


if __name__ == "__main__":
    init_llm_clients()

    TEST_PROMPT = """
    This is a test of your ability to respond to a request with JSON.  Please respond with a JSON object in the following format:
    {
        "key": "value",
        "key2": "value2"
    }
    Do not include any additional text in your response.
    """
    for client in ACTIVE_LLM_CLIENT_DICT.values():
        print("Testing " + client.model_name)
        try:
            response = client.ratellmiter_is_llm_blocked()
            print(client.model_name + "blocked: " + str(response))
            #response = client.prompt(str(uuid.uuid4()), TEST_PROMPT, json_output=True)
            #print(str(response.response_dict) + " input cost: " + str(response.input_cost) + " output cost: " + str(
            #    response.output_cost))
        except Exception as e:
            print("Exception: " + str(e))
            continue

        print("Prompt completed")
    print("All tests completed")
    exit(0)
