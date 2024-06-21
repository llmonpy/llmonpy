#   Copyright © 2024 Thomas Edward Burns
#  #
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following conditions:
#  #
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#   Software.
#  #
#   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import concurrent
import json

from jinja2 import Template
from nothingpy import Nothing

from llmonpy_step import LLMonPyStep, LLMonPyStepOutput
from llm_client import LlmClient
from trace_log import LlmClientInfo

DEFAULT_OUTPUT_DICT_KEY = "response_string"
TEMP_SETTING_KEY = "temp"


class LLMonPyPrompt:

    test_prompt_text: str

    class LLMonPyOutput(LLMonPyStepOutput):
        def __init__(self):
            pass

        @staticmethod
        def from_dict(dict):
            pass

    def __init__(self):
        if hasattr(self.__class__, "prompt_text") is False:
            raise AttributeError("Prompt text not defined in class")
        if hasattr(self.__class__, "json_output") is False:
            raise AttributeError("JSON output not defined in class")
        if hasattr(self.__class__, "LLMonPyOutput") is False:
            raise AttributeError("LLMonPyOutput not definedLLMonPyOutput in class")

    def get_prompt_text(self):
        return self.__class__.prompt_text

    def get_json_output(self):
        return self.__class__.json_output

    def get_step_name(self):
        result = self.__class__.__module__ + "." + self.__class__.__name__
        return result

    def to_dict(self):
        pass

    def from_dict(self, dict):
        pass

    def output_from_string(self, string):
        output_dict = { DEFAULT_OUTPUT_DICT_KEY: string }
        result = self.output_from_dict(output_dict)
        return result

    def output_from_dict(self, output_dict):
        result = self.LLMonPyOutput.from_dict(output_dict)
        return result


class FewShotPrompt:
    def set_example_list(self, example_list):
        pass


# make different evaluators if they handle errors different
class LLMonPyPromptEvaluator(LLMonPyStep):
    def __init__(self, llm_client: LlmClient, prompt: LLMonPyPrompt, temp: float = 0.0 ):
        self.llm_client = llm_client
        self.temp = temp
        self.prompt = prompt
        self.template = Template(prompt.get_prompt_text())

    def get_thread_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        result = self.llm_client.get_thread_pool()
        return result

    def set_example_list(self, example_list):
        if isinstance(self.prompt, FewShotPrompt):
            self.prompt.set_example_list(example_list)

    def get_prompt(self) -> LLMonPyPrompt:
        return self.prompt

    def get_step_name(self):
        return self.prompt.get_step_name()

    def get_input_dict(self):
        return self.prompt.to_dict()

    def get_llm_client_info(self):
        result = LlmClientInfo(self.llm_client.model_name, {TEMP_SETTING_KEY: self.temp})
        return result

    def execute_step(self, recorder):
        prompt_dict = self.prompt.to_dict()
        recorder.log_prompt_template(self.prompt.get_prompt_text())
        prompt_text = self.template.render(prompt_dict)
        result = None
        for i in range(0, 3):
            try:
                response = self.llm_client.prompt(prompt_text, Nothing, self.prompt.json_output,
                                                  self.temp)
                recorder.record_cost(response.get_response_cost())
                recorder.log_prompt_response(prompt_text, response.response_text)
                if self.prompt.json_output:
                    result = self.prompt.output_from_dict(response.response_dict)
                else:
                    result = self.prompt.output_from_string(response.response_text)
                break
            except Exception as e:
                recorder.log_exception(e)
                if i == 2:
                    raise e
                else:
                    continue
        return result, self


def create_prompt_steps(prompt, client_list, temp_list=None):
    result = []
    temp_list = temp_list if temp_list is not None else [0.0]
    for client in client_list:
        for temp in temp_list:
            result.append(LLMonPyPromptEvaluator(client, prompt, temp))
    return result


