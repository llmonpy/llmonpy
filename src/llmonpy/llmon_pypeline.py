import concurrent
import copy
import traceback

from llmonpy.llmonpy_step import LLMonPyStep, TraceLogRecorderInterface, STEP_TYPE_PYPELINE, \
    get_step_name_from_class_hierarchy, DEFAULT_TIMEOUT_TIME
from llmonpy.trace_log import trace_log_service


class LLMonPypeline:

    def __init__(self):
        pass

    def execute_step(self, recorder: TraceLogRecorderInterface):
        raise NotImplementedError()

    def get_step_type(self) -> str:
        return STEP_TYPE_PYPELINE

    def get_step_name(self):
        result = get_step_name_from_class_hierarchy(self.__class__)
        return result

    def get_short_step_name(self):
        result = self.__class__.__name__
        return result

    # recorder included as a parameter so that examples for ICL are part of input
    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        raise NotImplementedError()

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    def create_step(self, parent_recorder: TraceLogRecorderInterface):
        return LLMonPypelineRunner(parent_recorder, self)

    # step are returned in the order they complete, not the order they were submitted.  Also, if a step timesout, it
    # will be retried once.  If it fails again, it will not be included in the result list.
    # this does not work because the TimeoutError is caught at the highest level, not at the level the exception is
    # thrown.
    '''def run_parallel_steps_with_retry(self, step_list, timeout_time=DEFAULT_TIMEOUT_TIME, handle_result_function=None):
        future_list = []
        result_dict = {}
        timeout_exception = None
        for step in step_list:
            future = step.get_thread_pool().submit(step.record_step)
            future_list.append(future)
        for future in concurrent.futures.as_completed(future_list, timeout=timeout_time):
            try:
                returned_step = future.result()
                result_dict[returned_step.get_step_id()] = returned_step
                if handle_result_function is not None:
                    handle_result_function(returned_step)
            except TimeoutError as te:
                timeout_exception = te
                print("TimeoutError")
            except Exception as e:
                print(str(e)) # exception was logged in record_step
                pass
        if len(result_dict) < len(step_list):
            retry_list = []
            for step in step_list:
                if step.get_step_id() not in result_dict:
                    step.get_recorder().log_exception(timeout_exception)
                    retry_list.append(step)
            print("Retrying steps that timed out " + str(len(retry_list)))
            for step in retry_list:
                future = step.get_thread_pool().submit(step.retry_step)
                future_list.append(future)
            for future in concurrent.futures.as_completed(future_list, timeout=(timeout_time * 2)):
                try:
                    returned_step = future.result()
                    result_dict[returned_step.get_step_id()] = returned_step
                    if handle_result_function is not None:
                        handle_result_function(returned_step)
                except Exception as e:
                    print(str(e))
                    pass
        result_list = list(result_dict.values())
        return result_list'''

    def run_parallel_steps(self, step_list, handle_result_function=None):
        future_list = []
        result_dict = {}
        for step in step_list:
            future = step.get_thread_pool().submit(step.record_step)
            future_list.append(future)
        for future in concurrent.futures.as_completed(future_list):
            try:
                returned_step = future.result()
                result_dict[returned_step.get_step_id()] = returned_step
                if handle_result_function is not None:
                    handle_result_function(returned_step)
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(stack_trace)
                print(str(e)) # exception was logged in record_step
                pass
        result_list = list(result_dict.values())
        return result_list


class LLMonPypelineRunner(LLMonPyStep):
    def __init__(self, parent_recorder: TraceLogRecorderInterface, pypeline: LLMonPypeline):
        super().__init__()
        self.pypeline = pypeline
        if parent_recorder is None:
            self.recorder = trace_log_service().create_root_recorder(None, None, None, self)
        else:
            self.recorder = parent_recorder.create_child_recorder(self)

    def get_step_name(self):
        return self.pypeline.get_step_name()

    def get_step_type(self) -> str:
        return self.pypeline.get_step_type()

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        super_result = super().get_input_dict(recorder)
        result = self.pypeline.get_input_dict(recorder)
        result.update(super_result)
        return result

    def execute_step(self):
        result = None
        try:
            result = self.pypeline.execute_step(self.get_recorder())
        except Exception as e:
            self.get_recorder().log_exception(e)
            raise e
        return result

