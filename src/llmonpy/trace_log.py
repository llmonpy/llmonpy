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
import os
import traceback
from datetime import datetime
import json
import threading
import uuid

from llmonpy.config import llmonpy_config
from llmonpy.llmonpy_step import LLMonPyStepOutput, LLMONPY_OUTPUT_FORMAT_JSON, STEP_STATUS_NO_STATUS, STEP_STATUS_SUCCESS, \
    TraceLogRecorderInterface, TourneyResultInterface, JudgedOutput, LlmModelInfo
from llmonpy.llmonpy_trace_store import SqliteLLMonPyTraceStore
from llmonpy.system_services import system_services
from llmonpy.system_startup import llmonpy_start, llmonpy_stop

NO_VARIATION_OF_TRACE_ID = "NO_VARIATION_OF_TRACE_ID"

NO_OUTPUT_KEY = "no_output"
NO_OUTPUT_VALUE = "no_output"
NO_OUTPUT_DICT = {"output": "NO_OUTPUT_DICT"}


def is_no_output_dict(output_dict):
    result = output_dict is not None and output_dict == NO_OUTPUT_DICT
    return result


class TraceInfo:
    def __init__(self, trace_id, trace_group_id, variation_of_trace_id, title, start_time=None, end_time=None,
                 status_code: int = STEP_STATUS_NO_STATUS, cost=0.0):
        self.trace_id = trace_id
        self.trace_group_id = trace_group_id
        self.variation_of_trace_id = variation_of_trace_id if variation_of_trace_id is not None else NO_VARIATION_OF_TRACE_ID
        self.title = title
        self.start_time = start_time
        self.end_time = end_time
        self.status_code = status_code
        self.cost = cost

    def to_dict(self):
        result = copy.copy(vars(self))
        result["start_time"] = result["start_time"].isoformat() if result["start_time"] is not None else None
        result["end_time"] = result["end_time"].isoformat() if result["end_time"] is not None else None
        return result

    def to_json(self):
        result = json.dumps(self.to_dict())
        return result

    @staticmethod
    def from_dict(dict):
        if dict["start_time"] is not None and isinstance(dict["start_time"], str):
            dict["start_time"] = datetime.fromisoformat(dict["start_time"])
        if dict["end_time"] is not None and isinstance(dict["end_time"], str):
            dict["end_time"] = datetime.fromisoformat(dict["end_time"])
        result = TraceInfo(**dict)
        return result





class LLMonPyLogEvent:
    def __init__(self, event_id, trace_id, step_id, event_type: str, event_time: datetime):
        self.event_id = event_id
        self.trace_id = trace_id
        self.step_id = step_id
        self.event_type = event_type
        self.event_time = event_time

    def to_dict(self):
        result = copy.copy(vars(self))
        result["event_time"] = result["event_time"].isoformat()
        return result

    def to_json(self):
        result = json.dumps(self.to_dict())
        return result

    def after_from_dict(self):
        self.event_time = datetime.fromisoformat(self.event_time) if (self.event_time is not None and
                                                                      isinstance(self.event_time,str)) else self.event_time


class LLMonPyLogMessage(LLMonPyLogEvent):
    type_name = "message"

    def __init__(self, event_id, trace_id, step_id, event_time: datetime, message: str, event_type=None):
        super().__init__(event_id, trace_id, step_id, LLMonPyLogMessage.type_name, event_time)
        self.message = message

    @staticmethod
    def from_dict(dict):
        result = LLMonPyLogMessage(**dict)
        result.after_from_dict()
        return result

    @staticmethod
    def from_message(trace_id, step_id, message):
        event_id = str(uuid.uuid4())
        event_time = datetime.now()
        result = LLMonPyLogMessage(event_id, trace_id, step_id, event_time, message)
        return result


class LLMonPyLogException(LLMonPyLogEvent):
    type_name = "exception"

    def __init__(self, event_id, trace_id, step_id, event_time: datetime, exception_message: str,
                 stack_trace: str, event_type=None):
        super().__init__(event_id, trace_id, step_id, LLMonPyLogException.type_name, event_time)
        self.exception_message = exception_message
        self.stack_trace = stack_trace

    @staticmethod
    def from_dict(dict):
        result = LLMonPyLogException(**dict)
        result.after_from_dict()
        return result

    @staticmethod
    def from_exception(trace_id, step_id, exception):
        event_id = str(uuid.uuid4())
        event_time = datetime.now()
        stack_trace = traceback.format_exc()
        result = LLMonPyLogException(event_id, trace_id, step_id, event_time, str(exception),
                                     stack_trace)
        return result


class LLMonPyLogPromptTemplate(LLMonPyLogEvent):
    type_name = "prompt_template"

    def __init__(self, event_id, trace_id, step_id, event_time: datetime, prompt_template: str, event_type=None):
        super().__init__(event_id, trace_id, step_id, LLMonPyLogPromptTemplate.type_name, event_time)
        self.prompt_template = prompt_template

    @staticmethod
    def from_dict(dict):
        result = LLMonPyLogPromptTemplate(**dict)
        result.after_from_dict()
        return result

    @staticmethod
    def from_prompt_template(trace_id, step_id, prompt_template):
        event_id = str(uuid.uuid4())
        event_time = datetime.now()
        result = LLMonPyLogPromptTemplate(event_id, trace_id, step_id, event_time, prompt_template)
        return result


class LLMonPyLogPromptResponse(LLMonPyLogEvent):
    type_name = "prompt_response"

    def __init__(self, event_id, trace_id, step_id, event_time: datetime, prompt_text: str,
                 response_text: str, event_type=None):
        super().__init__(event_id, trace_id, step_id, LLMonPyLogPromptResponse.type_name, event_time)
        self.prompt_text = prompt_text
        self.response_text = response_text

    @staticmethod
    def from_dict(dict):
        result = LLMonPyLogPromptResponse(**dict)
        result.after_from_dict()
        return result

    @staticmethod
    def from_prompt_response(trace_id, step_id, prompt_text, response_text):
        event_id = str(uuid.uuid4())
        event_time = datetime.now()
        result = LLMonPyLogPromptResponse(event_id, trace_id, step_id, event_time, prompt_text, response_text)
        return result


class ContestResult:
    def __init__(self, contestant_one_output_id, contestant_two_output_id, winner_output_id, dissenting_judges:int = 0):
        self.contestant_one_output_id = contestant_one_output_id
        self.contestant_two_output_id = contestant_two_output_id
        self.winner_output_id = winner_output_id
        self.dissenting_judges = dissenting_judges

    def to_dict(self):
        result = copy.deepcopy(vars(self))
        return result

    def to_json(self):
        result_dict = self.to_dict()
        result = json.dumps(result_dict)
        return result

    @staticmethod
    def from_dict(dictionary):
        return ContestResult(**dictionary)


class TourneyResult(TourneyResultInterface):
    def __init__(self, tourney_result_id, step_id, trace_id, step_name, start_time, input_data, number_of_judges,
                 contestant_list: [LLMonPyStepOutput] = None,
                 contest_result_list: [ContestResult] = None):
        self.tourney_result_id = tourney_result_id
        self.step_id = step_id
        self.trace_id = trace_id
        self.step_name = step_name
        self.start_time = start_time
        self.input_data = input_data
        self.number_of_judges = number_of_judges
        self.contestant_list = contestant_list if contestant_list is not None else []
        self.contest_result_list = contest_result_list if contest_result_list is not None else []

    def add_contest_result(self, contestant_1_output_id, contestant_2_output_id, winner_output_id, dissenting_judges=0):
        result = ContestResult(contestant_1_output_id, contestant_2_output_id, winner_output_id, dissenting_judges)
        self.contest_result_list.append(result)

    def to_dict(self):
        result = copy.deepcopy(vars(self))
        result["start_time"] = result["start_time"].isoformat() if result["start_time"] is not None else None
        result["contestant_list"] = [contestant.to_dict() for contestant in result["contestant_list"]]
        result["contest_result_list"] = [contest_result.to_dict() for contest_result in result["contest_result_list"]]
        return result

    def to_json(self):
        result_dict = self.to_dict()
        result = json.dumps(result_dict)
        return result

    @staticmethod
    def from_dict(dictionary):
        tourney_result = TourneyResult(**dictionary)
        tourney_result.contestant_list = [JudgedOutput.from_dict(contestant) for contestant in tourney_result.contestant_list] if tourney_result.contestant_list is not None else []
        tourney_result.contest_result_list = [ContestResult.from_dict(contest_result) for contest_result in tourney_result.contest_result_list] if tourney_result.contest_result_list is not None else []
        if tourney_result.start_time is not None and isinstance(tourney_result.start_time, str):
            tourney_result.start_time = datetime.fromisoformat(tourney_result.start_time)
        return tourney_result


class StepTraceData:
    def __init__(self, trace_id, trace_group_id, variation_of_trace_id, step_id, step_index,
                 step_name, step_type, root_step_id, root_step_name, parent_step_id, parent_step_name, llm_model_info, input_dict,
                 start_time=None, end_time=None, output_dict=None, output_format=LLMONPY_OUTPUT_FORMAT_JSON,
                 status_code=STEP_STATUS_NO_STATUS, error_list=None, cost=0.0):
        self.trace_id = trace_id
        self.trace_group_id = trace_group_id
        self.variation_of_trace_id = variation_of_trace_id
        self.step_id = step_id
        self.step_index = step_index
        self.llm_model_info = llm_model_info
        self.step_name = step_name
        self.step_type = step_type
        self.root_step_id = root_step_id
        self.root_step_name = root_step_name
        self.parent_step_id = parent_step_id
        self.parent_step_name = parent_step_name
        self.input_dict = input_dict
        self.start_time = start_time
        self.end_time = end_time
        self.output_dict = output_dict
        self.output_format = output_format
        self.status_code = status_code
        self.error_list = error_list
        self.cost = cost

    def to_dict(self):
        result = copy.copy(vars(self))
        if result["llm_model_info"] is not None:
            result["llm_model_info"] = result["llm_model_info"].to_dict()
        result["start_time"] = result["start_time"].isoformat() if result["start_time"] is not None else None
        result["end_time"] = result["end_time"].isoformat() if result["end_time"] is not None else None
        return result

    def to_json(self):
        result_dict = self.to_dict()
        result = json.dumps(result_dict)
        return result

    def add_exception(self, exception):
        if self.error_list is None:
            self.error_list = []
        self.error_list.append(str(exception))

    @staticmethod
    def from_dict(dict):
        result = StepTraceData(**dict)
        if result.llm_model_info is not None:
            result.llm_model_info = LlmModelInfo.from_dict(result.llm_model_info)
        if result.start_time is not None and isinstance(result.start_time, str):
            result.start_time = datetime.fromisoformat(result.start_time)
        if result.end_time is not None and isinstance(result.end_time, str):
            result.end_time = datetime.fromisoformat(result.end_time)
        return result


class CompleteTraceData:
    def __init__(self, trace_info: TraceInfo, step_list: [StepTraceData], tourney_result_list: [TourneyResult]):
        self.trace_info = trace_info
        self.step_list = step_list
        self.tourney_result_list = tourney_result_list

    def to_dict(self):
        result = copy.copy(vars(self))
        result["trace_info"] = self.trace_info.to_dict()
        result["step_list"] = [step.to_dict() for step in self.step_list]
        result["tourney_result_list"] = [tourney_result.to_dict() for tourney_result in self.tourney_result_list]
        return result

    def to_json(self):
        result_dict = self.to_dict()
        result = json.dumps(result_dict)
        return result


class TraceLogRecorder (TraceLogRecorderInterface):
    def __init__(self, trace_log_service, root_recorder, parent_recorder, trace_id, trace_group_id,
                 variation_of_trace_id, step_id, step_index, step, root_step_id, root_step_name, parent_step_id,
                 parent_step_name, client_info, input_dict=None, start_time=None):
        self.trace_log_service = trace_log_service
        self.step = step
        self.trace_data = StepTraceData(trace_id, trace_group_id, variation_of_trace_id, step_id, step_index,
                                        step.get_step_name(), step.get_step_type(),
                                        root_step_id, root_step_name, parent_step_id, parent_step_name, client_info,
                                        input_dict, start_time)
        self.root_recorder = root_recorder
        self.parent_recorder = parent_recorder
        self.recorder_lock = threading.Lock()
        self.next_step_index = step_index
        self.step_examples = {}

    def get_step_id(self):
        return self.trace_data.step_id

    def get_trace_id(self):
        return self.trace_data.trace_id

    def get_model_info(self) -> LlmModelInfo:
        return self.trace_data.llm_model_info

    def get_input_dict(self):
        return self.trace_data.input_dict

    def get_next_step_index(self):
        if self.root_recorder is not None:
            result = self.root_recorder.get_next_step_index()
        else:
            with self.recorder_lock:
                self.next_step_index += 1
                result = self.next_step_index
        return result

    def set_step_examples(self, step_name: str, example_list: [LLMonPyStepOutput]):
        self.step_examples[step_name] = example_list

    def get_step_examples(self, step_name: str) -> [LLMonPyStepOutput]:
        result = self.step_examples.get(step_name, None)
        if result is None and self.parent_recorder is not None:
            result = self.parent_recorder.get_step_examples(step_name)
        return result

    def log_message(self, message):
        event = LLMonPyLogMessage.from_message(self.trace_data.trace_id, self.trace_data.step_id, message)
        self.trace_log_service.record_event(event)

    def log_exception(self, exception):
        event = LLMonPyLogException.from_exception(self.trace_data.trace_id, self.trace_data.step_id, exception)
        self.trace_log_service.record_event(event)

    def log_prompt_template(self, prompt_template):
        event = LLMonPyLogPromptTemplate.from_prompt_template(self.trace_data.trace_id, self.trace_data.step_id,
                                                              prompt_template)
        self.trace_log_service.record_event(event)

    def log_prompt_response(self, prompt_text, response_text):
        event = LLMonPyLogPromptResponse.from_prompt_response(self.trace_data.trace_id, self.trace_data.step_id,
                                                              prompt_text, response_text)
        self.trace_log_service.record_event(event)

    def add_to_cost(self, cost):
        with self.recorder_lock:
            self.trace_data.cost += cost

    def create_child_recorder(self, step):
        input_dict = step.get_input_dict(self)
        client_info = step.get_llm_model_info()
        step_id = str(uuid.uuid4())
        step_index = self.get_next_step_index()
        start_time = datetime.now()
        root_recorder = self.root_recorder if self.root_recorder is not None else self
        result = TraceLogRecorder(self.trace_log_service, root_recorder, self, self.trace_data.trace_id,
                                  self.trace_data.trace_group_id, self.trace_data.variation_of_trace_id, step_id,
                                  step_index, step, self.trace_data.root_step_id,
                                  self.trace_data.root_step_name, self.trace_data.step_id, self.trace_data.step_name,
                                  client_info, input_dict, start_time)
        return result

    def record_exception(self, exception):
        if exception is not None:
            self.trace_data.add_exception(exception)
            self.log_exception(exception)

    def record_cost(self, cost):
        if cost is not None:
            self.add_to_cost(cost)
            if self.parent_recorder is not None:
                self.parent_recorder.record_cost(cost)

    def create_tourney_result(self, number_of_judges, judged_step_name) -> TourneyResult:
        tourney_result_id = str(uuid.uuid4())
        result = TourneyResult(tourney_result_id, self.trace_data.step_id, self.trace_data.trace_id,
                               judged_step_name, self.trace_data.start_time, self.trace_data.input_dict, number_of_judges)
        return result

    def record_tourney_result(self, contestant_list: [LLMonPyStepOutput], tourney_result):
        contestant_list = copy.deepcopy(contestant_list)
        tourney_result.contestant_list = contestant_list
        self.trace_log_service.record_tourney_result(tourney_result)
           
    def finish_child_step(self, output_dict, status_code=STEP_STATUS_SUCCESS,
                          cost=None):
        if cost is not None:
            self.add_to_cost(cost)
        end_time = datetime.now()
        self.trace_data.end_time = end_time
        if isinstance(output_dict, list):
            dict_list = [output.to_dict() for output in output_dict]
            list_output = {"output_list": dict_list}
            self.trace_data.output_dict = list_output
        else:
            output_dict = output_dict.to_dict() if output_dict is not None else None
            self.trace_data.output_dict = output_dict
        self.trace_data.output_format = self.step.get_output_format()
        self.trace_data.status_code = status_code
        self.trace_log_service.record_step(self.trace_data)
        if self.parent_recorder is None:
            trace_info = TraceInfo(self.trace_data.trace_id, self.trace_data.trace_group_id, self.trace_data.variation_of_trace_id,
                      self.trace_data.step_name, self.trace_data.start_time, end_time, status_code, self.trace_data.cost)
            self.trace_log_service.flush_trace(trace_info)


class TraceLogService:
    def __init__(self):
        self.data_directory = llmonpy_config().data_directory
        self.step_file_path = os.path.join(self.data_directory, "steps.jsonl")
        self.events_file_path = os.path.join(self.data_directory, "events.jsonl")
        self.tourney_result_file_path = os.path.join(self.data_directory, "tourney_results.jsonl")
        self.trace_info_file_path = os.path.join(self.data_directory, "trace_info.jsonl")
        self.llmonpy_trace_store: SqliteLLMonPyTraceStore = SqliteLLMonPyTraceStore(self.data_directory,
                                                                                    TraceInfo.from_dict,
                                                                                    StepTraceData.from_dict,
                                                                                    self.event_from_dict,
                                                                                    TourneyResult.from_dict)
        self.event_factory_dict = {}
        self.init_event_factory()
        self.write_lock = threading.Lock()
        self.recorded_step_list = []
        self.event_list = []
        self.tourney_result_list = []
        self.trace_info_list = []


    def init_event_factory(self):
        subclasses = LLMonPyLogEvent.__subclasses__()
        for subclass in subclasses:
            self.event_factory_dict[subclass.type_name] = subclass.from_dict

    def event_from_dict(self, dict):
        event_type = dict["event_type"]
        factory = self.event_factory_dict[event_type]
        result = factory(dict)
        return result

    def stop(self):
        self.llmonpy_trace_store.stop()

    def create_root_recorder(self, trace_id, trace_group_id, variation_of_trace_id, step) -> TraceLogRecorder:
        root_step_id = str(uuid.uuid4())
        start_time = datetime.now()
        input_dict = step.get_input_dict(None)
        step_name = step.get_step_name()
        client_info = step.get_llm_model_info()
        result = TraceLogRecorder(self, None, None, trace_id, trace_group_id,
                                  variation_of_trace_id, root_step_id, 0, step, root_step_id, step_name,
                                  None, None, client_info, input_dict, start_time)
        return result

    def flush_trace(self, trace_info):
        with self.write_lock:
            self.trace_info_list.append(trace_info)
        self.write_data()

    def record_step(self, trace_data):
        with self.write_lock:
            self.recorded_step_list.append(trace_data)

    def record_event(self, event):
        with self.write_lock:
            self.event_list.append(event)

    def record_tourney_result(self, tourney_result):
        with self.write_lock:
            self.tourney_result_list.append(tourney_result)

    def get_and_clear_recorded_steps(self):
        with self.write_lock:
            result = self.recorded_step_list
            self.recorded_step_list = []
        return result

    def get_and_clear_recorded_events(self):
        with self.write_lock:
            result = self.event_list
            self.event_list = []
        return result

    def get_and_clear_recorded_tourney_results(self):
        with self.write_lock:
            result = self.tourney_result_list
            self.tourney_result_list = []
        return result

    def get_and_clear_recorded_trace_info(self):
        with self.write_lock:
            result = self.trace_info_list
            self.trace_info_list = []
        return result

    def write_data(self):
        steps_ready_to_write = self.get_and_clear_recorded_steps()
        if len(steps_ready_to_write) > 0:
            with open(self.step_file_path, "a") as file:
                for step in steps_ready_to_write:
                    file.write(step.to_json() + "\n")
            self.llmonpy_trace_store.insert_step_records(steps_ready_to_write)
        events_ready_to_write = self.get_and_clear_recorded_events()
        if len(events_ready_to_write) > 0:
            with open(self.events_file_path, "a") as file:
                for event in events_ready_to_write:
                    file.write(event.to_json() + "\n")
            self.llmonpy_trace_store.insert_events(events_ready_to_write)
        tourney_results_ready_to_write = self.get_and_clear_recorded_tourney_results()
        if len(tourney_results_ready_to_write) > 0:
            with open(self.tourney_result_file_path, "a") as file:
                for tourney_result in tourney_results_ready_to_write:
                    file.write(tourney_result.to_json() + "\n")
            self.llmonpy_trace_store.insert_tourney_results(tourney_results_ready_to_write)
        trace_info_ready_to_write = self.get_and_clear_recorded_trace_info()
        if len(trace_info_ready_to_write) > 0:
            with open(self.trace_info_file_path, "a") as file:
                for trace_info in trace_info_ready_to_write:
                    file.write(trace_info.to_json() + "\n")
            self.llmonpy_trace_store.insert_trace_info(trace_info_ready_to_write)

    def get_trace_list(self) -> [TraceInfo]:
        result = self.llmonpy_trace_store.get_trace_list()
        if result is not None and len(result) > 0:
            result.sort(key=lambda x: x.start_time, reverse=True)
        return result

    def get_steps_for_trace(self, trace_id):
        result = self.llmonpy_trace_store.get_steps_for_trace(trace_id)
        result.sort(key=lambda x: x.step_index)
        return result

    def get_events_for_step(self, step_id):
        result = self.llmonpy_trace_store.get_events_for_step(step_id)
        result.sort(key=lambda x: x.event_time)
        return result

    def get_complete_trace_by_id(self, trace_id: str) -> CompleteTraceData:
        trace_info = self.llmonpy_trace_store.get_trace_by_id(trace_id)
        step_list = self.llmonpy_trace_store.get_steps_for_trace(trace_id)
        step_list.sort(key=lambda x: x.step_index)
        tourney_result_list = self.llmonpy_trace_store.get_tourney_results_for_trace(trace_id)
        tourney_result_list.sort(key=lambda x: x.start_time)
        result = CompleteTraceData(trace_info, step_list, tourney_result_list)
        return result

    def get_tourney_step_name_list(self) -> [str]:
        result = self.llmonpy_trace_store.get_tourney_step_name_list()
        return result

    def get_tourney_results_for_step_name(self, step_name: str) -> [TourneyResult]:
        result = self.llmonpy_trace_store.get_tourney_results_for_step_name(step_name)
        result.sort(key=lambda x: x.start_time, reverse=True)
        return result


def init_trace_log_service():
    result = TraceLogService()
    system_services().set_trace_log_service(result)


def trace_log_service() -> TraceLogService:
    result = system_services().trace_log_service
    return result


if __name__ == "__main__":
    llmonpy_start()
    print("Running Test trace_log.py")
    try:
        trace_list = trace_log_service().get_trace_list()
        for main_trace_info in trace_list:
            print("id: " + main_trace_info.trace_id + " title: " + main_trace_info.title + " start: " + str(main_trace_info.start_time) +
                    " cost: " + str(main_trace_info.cost))
        main_trace_id = trace_list[0].trace_id
        complete_trace = trace_log_service().get_complete_trace_by_id(main_trace_id)
        tourney_step_name_list = trace_log_service().get_tourney_step_name_list()
        print("complete trace: " + complete_trace.to_json())
        print("tourney step names: " + str(tourney_step_name_list))
        for step_name in tourney_step_name_list:
            tourney_results = trace_log_service().get_tourney_results_for_step(step_name)
            print("tourney results for " + step_name)
            for tourney_result in tourney_results:
                print(tourney_result.to_json())
    except Exception as e:
        stack_trace = traceback.format_exc()
        error_message = str(e) + " " + stack_trace
        print(error_message)
    llmonpy_stop()
    exit(0)
