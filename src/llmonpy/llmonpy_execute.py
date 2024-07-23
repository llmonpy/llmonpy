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
import uuid

from llmonpy.llmonpy_step import LLMonPyStep, STEP_STATUS_FAILURE, TraceLogRecorderInterface
from llmonpy.trace_log import trace_log_service


class FutureStepList:
    class FutureStep:
        def __init__(self, future, recorder: TraceLogRecorderInterface):
            self.future = future
            self.recorder = recorder

    def __init__(self):
        self.future_step_list = []

    def add_future_step(self, future_step):
        self.future_step_list.append(future_step)

    def get_future_list(self):
        result = [future_step.future for future_step in self.future_step_list]
        return result

    def get_step_id_list(self):
        result = [future_step.recorder.step_id for future_step in self.future_step_list]
        return result

    def wait(self):
        future_list = self.get_future_list()
        step_id_list = self.get_step_id_list()
        result_list = [None for _ in range(len(step_id_list))]
        for future in concurrent.futures.as_completed(future_list):
            try:
                output, recorder = future.result()
                for i, step_id in enumerate(result_list):
                    if step_id == recorder.get_step_id():
                        result_list[i] = output
                        break
            except Exception as e:
                print(str(e))
                pass
        return result_list


def do_llmonpy_step(step: LLMonPyStep, recorder: TraceLogRecorderInterface, child_recorder=None):
    child_recorder = recorder.create_child_recorder(step) if child_recorder is None else child_recorder
    try:
        result, recorder_result = step.execute_step(child_recorder)
        child_recorder.finish_child_step(result)
    except Exception as e:
        child_recorder.record_exception(e)
        child_recorder.finish_child_step(None, status_code=STEP_STATUS_FAILURE)
        raise e
    return result, recorder_result


def do_llmonpy_parallel_step(step: LLMonPyStep, recorder: TraceLogRecorderInterface):
    child_recorder = recorder.create_child_recorder(step)
    future = step.get_thread_pool().submit(do_llmonpy_step, step, recorder, child_recorder)
    return future, child_recorder


def do_step_list_wait(step_list:[LLMonPyStep], recorder: TraceLogRecorderInterface):
    future_step_list = do_step_list_no_wait(step_list, recorder)
    result = future_step_list.wait()
    return result

def do_step_list_no_wait(step_list:[LLMonPyStep], recorder: TraceLogRecorderInterface):
    result = FutureStepList()
    for step in step_list:
        future, child_recorder = do_llmonpy_parallel_step(step, recorder)
        result.add_future_step(FutureStepList.FutureStep(future, child_recorder))
    return result


def run_step(step):
    trace_id = str(uuid.uuid4())
    result = None
    try:
        recorder = trace_log_service().create_root_recorder(trace_id, trace_id, None, step)
        result, _ = step.execute_step(recorder)
    finally:
        recorder.finish_child_step(result)
    return result, recorder