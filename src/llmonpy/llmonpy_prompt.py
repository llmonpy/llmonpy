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

from llmonpy.llmonpy_step import *
from llmonpy.llm_client import LlmClient
from llmonpy.trace_log import LlmModelInfo

DEFAULT_OUTPUT_DICT_KEY = "response_string"


class LLMonPyPrompt(LLMonPyStep):
    class LLMonPyOutput(LLMonPyStepOutput):
        def __init__(self):
            pass

        @staticmethod
        def from_dict(dict):
            pass

    def __init__(self):
        if hasattr(self.__class__, "prompt_text") is False:
            raise AttributeError("prompt_text not defined in class")
        if hasattr(self.__class__, "output_format") is False:
            raise AttributeError("output_format not defined in class")
        if hasattr(self.__class__, "LLMonPyOutput") is False:
            raise AttributeError("LLMonPyOutput not definedLLMonPyOutput in class")

    def get_prompt_text(self):
        return self.__class__.prompt_text

    def get_json_output(self):
        return self.__class__.output_format == LLMONPY_OUTPUT_FORMAT_JSON

    def get_output_format(self):
        result = self.__class__.output_format
        return result

    def get_step_name(self):
        result = get_step_name_from_class_hierarchy(self.__class__)
        return result

    def to_dict(self):
        result = copy.deepcopy(vars(self))
        return result

    def from_dict(self, dict):
        pass

    def output_from_string(self, string):
        output_dict = { DEFAULT_OUTPUT_DICT_KEY: string }
        result = self.output_from_dict(output_dict)
        return result

    def output_from_dict(self, output_dict):
        result = self.LLMonPyOutput.from_dict(output_dict)
        return result


# make different evaluators if they handle errors different
class LLMonPyPromptExecutor(LLMonPyStep):
    def __init__(self, prompt: LLMonPyPrompt, llm_model_info: LlmModelInfo):
        self.llm_model_info = llm_model_info
        self.prompt = prompt
        self.template = Template(prompt.get_prompt_text())

    def get_thread_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        result = self.get_llm_client().get_thread_pool()
        return result

    def get_prompt(self) -> LLMonPyPrompt:
        return self.prompt

    def get_step_name(self):
        return self.prompt.get_step_name()

    def get_step_type(self) -> str:
        return STEP_TYPE_PROMPT

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        super_result = super().get_input_dict(recorder)
        result = self.prompt.to_dict()
        result.update(super_result)
        return result

    def get_llm_client(self) -> LlmClient:
        result = self.llm_model_info.get_llm_client()
        return result

    def get_llm_model_info(self):
        result = self.llm_model_info
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        prompt_dict = recorder.get_input_dict()
        recorder.log_prompt_template(self.prompt.get_prompt_text())
        prompt_text = self.template.render(prompt_dict)
        result = None
        for i in range(0, 3):
            try:
                response = self.get_llm_client().prompt(prompt_text, Nothing, self.prompt.get_json_output(),
                                                  self.llm_model_info.get_temp())
                recorder.record_cost(response.get_response_cost())
                recorder.log_prompt_response(prompt_text, response.response_text)
                if self.prompt.get_json_output():
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
        return result, recorder


def create_prompt_steps(prompt, model_info_list: [LlmModelInfo]):
    result = []
    for model_info in model_info_list:
        result.append(LLMonPyPromptExecutor(prompt, model_info))
    return result


