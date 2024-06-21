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

from llmonpy_step import LLMonPyStep
from trace_log import TraceLogRecorder, STEP_STATUS_FAILURE


def do_llmonpy_step(step: LLMonPyStep, recorder: TraceLogRecorder):
    child_recorder = recorder.create_child_recorder(step)
    try:
        result,_ = step.execute_step(child_recorder)
        child_recorder.finish_child_step(result)
    except Exception as e:
        child_recorder.record_exception(e)
        child_recorder.finish_child_step(None, status_code=STEP_STATUS_FAILURE)
        raise e
    return result, step


def do_llmonpy_parallel_step(step: LLMonPyStep, recorder: TraceLogRecorder ):
    future = step.get_thread_pool().submit(do_llmonpy_step, step, recorder)
    return future