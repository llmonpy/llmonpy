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

from config import llmonpy_config


LLMONPY_OUTPUT_FORMAT_JSON = "json"
LLMONPY_OUTPUT_FORMAT_TEXT = "text"

STEP_STATUS_NO_STATUS = 0
STEP_STATUS_SUCCESS = 200
STEP_STATUS_FAILURE = 500


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


class TourneyResultInterface:
    def add_contest_result(self, contestant_1_output_id, contestant_2_output_id, winner_output_id, dissenting_judges=0):
        raise NotImplementedError()


class TraceLogRecorderInterface:

    def get_step_id(self):
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

    def create_tourney_result(self, number_of_judges) -> TourneyResultInterface:
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

    def set_example_list(self, example_list):
        pass

    def get_step_name(self):
        result = self.__class__.__module__ + "." + self.__class__.__name__
        return result

    def get_input_dict(self):
        return {}

    def get_llm_client_info(self):
        return None

    def get_output_format(self):
        return LLMONPY_OUTPUT_FORMAT_JSON




