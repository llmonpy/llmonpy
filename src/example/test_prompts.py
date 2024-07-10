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
import uuid

from llm_client import MISTRAL_7B, TOGETHER_QWEN1_5_4B, TOGETHER_LLAMA3_70B
from prompt import LLMonPyPrompt, LLMonPyPromptEvaluator
from system_startup import system_startup, system_stop
from trace_log import trace_log_service


class TestLLMonPyPrompt(LLMonPyPrompt):
    prompt_text = """
            Hi! My name is {{ user_name }}.  What is my name?  Reply with JSON in the form { "name": "my name" }.
            For example, if my name is "Alice", you would reply with { "name": "Alice" }. Do not include any other
            text in your response.
            """
    json_output = True

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
    system_startup()
    try:
        trace_id = str(uuid.uuid4())
        print("Running TestLLMonPyPrompt")
        step = LLMonPyPromptEvaluator(MISTRAL_7B, TestLLMonPyPrompt("Tom"))
        recorder = trace_log_service().create_root_recorder(trace_id, trace_id, None, step)
        result, _ = step.execute_step(recorder)
        recorder.finish_child_step(result)
        print(result.to_json())
        time.sleep(6)
        step_list = trace_log_service().get_steps_for_trace(trace_id)
        for step in step_list:
            print(step.to_json())
            log_list = trace_log_service().get_events_for_step(step.step_id)
            print("Logs for step " + step.step_id + ":")
            for log in log_list:
                print(log.to_json())
    except Exception as e:
        print(str(e))
    finally:
        system_stop()
        exit(0)
