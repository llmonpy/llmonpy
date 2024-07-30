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
import copy
import time
import traceback
import uuid

from llmonpy.llm_client import MISTRAL_7B, TOGETHER_QWEN1_5_4B, TOGETHER_LLAMA3_70B, FIREWORKS_LLAMA3_1_8B, \
    FIREWORKS_LLAMA3_1_405B, FIREWORKS_LLAMA3_1_70B, FIREWORKS_GEMMA2_9B, FIREWORKS_MYTHOMAXL2_13B, FIREWORKS_QWEN2_72B
from llmonpy.llmonpy_execute import run_step
from llmonpy.llmonpy_prompt import LLMonPyPrompt, LLMonPyPromptExecutor
from llmonpy.system_startup import llmonpy_start, llmonpy_stop
from llmonpy.trace_log import trace_log_service
from llmonpy.llmonpy_step import LLMONPY_OUTPUT_FORMAT_JSON, LlmModelInfo


class TestLLMonPyPrompt(LLMonPyPrompt):
    prompt_text = """
            Hi! My name is {{ user_name }}.  What is my name?  Reply with JSON in the form { "name": "my name" }.
            For example, if my name is "Alice", you would reply with { "name": "Alice" }. Do not include any other
            text in your response.
            """
    output_format = LLMONPY_OUTPUT_FORMAT_JSON

    class LLMonPyOutput(LLMonPyPrompt.LLMonPyOutput):
        def __init__(self, name=None):
            super().__init__()
            self.name = name

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            return result

        @staticmethod
        def from_dict(dictionary):
            result = TestLLMonPyPrompt.LLMonPyOutput(**dictionary)
            return result

    def __init__(self, user_name):
        super().__init__()
        self.user_name = user_name

    def to_dict(self):
        result = copy.deepcopy(vars(self))
        return result

    def from_dict(self, dict):
        pass


if __name__ == "__main__":
    llmonpy_start()
    try:
        print("Running TestLLMonPyPrompt")
        model_info = LlmModelInfo(FIREWORKS_QWEN2_72B.model_name)
        step = LLMonPyPromptExecutor(TestLLMonPyPrompt("LLMonPy"), model_info)
        result, recorder = run_step(step)
        trace_id = recorder.get_trace_id()
        print(result.to_json())
        step_list = trace_log_service().get_steps_for_trace(trace_id)
        for step in step_list:
            print(step.to_json())
            log_list = trace_log_service().get_events_for_step(step.step_id)
            print("Logs for step " + step.step_id + ":")
            for log in log_list:
                print(log.to_json())
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(stack_trace)
        print(str(e))
    finally:
        llmonpy_stop()
        exit(0)
