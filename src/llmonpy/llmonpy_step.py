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
import copy
import json
import uuid

from llmonpy.llm_client import get_llm_client, LlmClient, filter_clients_that_didnt_start
from llmonpy.config import llmonpy_config


LLMONPY_OUTPUT_FORMAT_JSON = "json"
LLMONPY_OUTPUT_FORMAT_TEXT = "text"
EXAMPLE_LIST_KEY = "example_list"
STEP_NAME_SEPARATOR = ":"
TEMP_SETTING_KEY = "temp"

STEP_TYPE_PROMPT = "prompt"
STEP_TYPE_TOURNEY = "tourney"
STEP_TYPE_CYCLE = "cycle"
STEP_TYPE_PYPELINE = "pypeline"
STEP_TYPE_JUDGE = "judge"
STEP_TYPE_JURY = "jury"
STEP_TYPE_RANKER = "ranker"
STEP_TYPE_GENERATOR = "generator"
STEP_TYPE_GAR = "gar"

STEP_STATUS_NO_STATUS = 0
STEP_STATUS_SUCCESS = 200
STEP_STATUS_FAILURE = 500


def class_has_no_superclass(class_obj):
    return class_obj.__bases__ == (object,)


def get_step_name_from_class_hierarchy(class_obj):
    result = "" if class_has_no_superclass(class_obj) else (get_step_name_from_class_hierarchy(class_obj.__bases__[0])
                                                            + STEP_NAME_SEPARATOR)
    result += class_obj.__module__ + "." + class_obj.__name__
    return result


class LlmModelInfo:
    def __init__(self, model_name, client_settings_dict=None):
        self.model_name = model_name
        self.client_settings_dict = client_settings_dict if client_settings_dict is not None else {TEMP_SETTING_KEY: 0.0}

    def get_llm_client(self):
        result = get_llm_client(self.model_name)
        return result

    def get_model_name(self):
        return self.model_name

    def get_temp(self):
        result = self.client_settings_dict.get(TEMP_SETTING_KEY, None)
        return result

    def to_dict(self):
        result = copy.deepcopy(vars(self))
        return result

    def to_json(self):
        result = json.dumps(self.to_dict())
        return result

    @staticmethod
    def from_dict(dict):
        return LlmModelInfo(**dict)


class ModelTemp:
    def __init__(self, model_client_list: [LlmClient], temp):
        self.model_client_list = model_client_list
        if isinstance(temp, list):
            self.temp_list = temp
        else:
            self.temp_list = [temp]

    def get_model_info_list(self):
        result = []
        filtered_list = filter_clients_that_didnt_start(self.model_client_list)
        for model_client in filtered_list:
            for temp in self.temp_list:
                result.append(LlmModelInfo(model_client.model_name, {TEMP_SETTING_KEY: temp}))
        return result


def make_model_list(*args) -> [LlmModelInfo]:
    result = []
    for arg in args:
        arg_model_list = arg.get_model_info_list()
        result.extend(arg_model_list)
    return result


class LLMonPyStepOutput:
    def __init__(self):
        pass

    def __str__(self):
        result = json.dumps(self.to_dict())
        return result

    def to_dict(self):
        pass

    def to_json(self):
        result = json.dumps(self.to_dict())
        return result

    @staticmethod
    def from_dict(dict):
        pass


class DictLLMonPyStepOutput(LLMonPyStepOutput):
    def __init__(self, output_dict):
        self.output_dict = output_dict

    def to_dict(self):
        return self.output_dict

    @staticmethod
    def from_dict(dict):
        result = DictLLMonPyStepOutput(dict)
        return result


class JudgedOutput(LLMonPyStepOutput):
    def __init__(self, step_id=None, step_output=None, llm_model_info=None, output_id=None, victory_count=0):
        self.step_id = step_id
        self.output_id = str(uuid.uuid4()) if output_id is None else output_id
        self.llm_model_info = llm_model_info
        self.step_output = step_output
        self.victory_count = victory_count

    def reset_victory_count(self):
        self.victory_count = 0

    def to_dict(self):
        result = copy.deepcopy(vars(self))
        result["step_output"] = self.step_output.to_dict()
        if result["llm_model_info"] is not None:
            result["llm_model_info"] = result["llm_model_info"].to_dict()
        return result

    @staticmethod
    def from_dict(dictionary):
        dictionary["step_output"] = DictLLMonPyStepOutput.from_dict(dictionary["step_output"])
        dictionary["llm_model_info"] = LlmModelInfo.from_dict(dictionary["llm_model_info"])
        result = JudgedOutput(**dictionary)
        return result


class TourneyResultInterface:
    def add_contest_result(self, contestant_1_output_id, contestant_2_output_id, winner_output_id, dissenting_judges=0):
        raise NotImplementedError()


class TraceLogRecorderInterface:

    def get_step_id(self):
        raise NotImplementedError()

    def get_trace_id(self):
        raise NotImplementedError()

    def get_model_info(self) -> LlmModelInfo:
        raise NotImplementedError()

    def get_input_dict(self):
        raise NotImplementedError()

    def set_step_examples(self, step_name: str, example_list: [LLMonPyStepOutput]):
        raise NotImplementedError()

    def get_step_examples(self, step_name: str) -> [LLMonPyStepOutput]:
        raise NotImplementedError()

    def log_message(self, message):
        raise NotImplementedError()

    def log_exception(self, exception):
        raise NotImplementedError()

    def log_prompt_template(self, prompt_template):
        raise NotImplementedError()

    def log_prompt_response(self, prompt_text, response_text):
        raise NotImplementedError()

    def add_to_cost(self, cost):
        raise NotImplementedError()

    def create_child_recorder(self, step):
        raise NotImplementedError()

    def record_exception(self, exception):
        raise NotImplementedError()

    def record_cost(self, cost):
        raise NotImplementedError()

    def create_tourney_result(self, number_of_judges, judge_step_name) -> TourneyResultInterface:
        raise NotImplementedError()

    def record_tourney_result(self, contestant_list: [LLMonPyStepOutput], tourney_result):
        raise NotImplementedError()

    def finish_child_step(self, output_dict, status_code=STEP_STATUS_SUCCESS,
                          cost=None):
        raise NotImplementedError()


class LLMonPyStep:
    def execute_step(self, recorder: TraceLogRecorderInterface) -> (LLMonPyStepOutput, TraceLogRecorderInterface):
        raise NotImplementedError()

    def get_thread_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        return llmonpy_config().thread_pool

    def get_step_name(self):
        result = get_step_name_from_class_hierarchy(self.__class__)
        return result

    def get_short_step_name(self):
        result = self.__class__.__name__
        return result

    def get_step_type(self) -> str:
        raise NotImplementedError()

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        example_list = recorder.get_step_examples(self.get_step_name()) if recorder is not None else None
        if example_list is not None:
            example_list = [example.to_dict() for example in example_list]
            result = {EXAMPLE_LIST_KEY: example_list}
        else:
            result = {}
        return result

    def get_llm_model_info(self):
        return None

    def get_output_format(self):
        return LLMONPY_OUTPUT_FORMAT_JSON





