import concurrent
import traceback

from llm_client import FIREWORKS_LLAMA3_1_8B, MISTRAL_7B
from llmonpy_prompt import LLMonPyPromptRunner
from llmonpy_step import LlmModelInfo
from system_startup import llmonpy_start, llmonpy_stop
from test_prompts import TestLLMonPyPrompt
from trace_log import trace_log_service

if __name__ == "__main__":
    llmonpy_start()
    try:
        print("Running Test Rate Limiter")
        step_list = []
        for i in range(200):
            model_info = LlmModelInfo(MISTRAL_7B.model_name)
            step = LLMonPyPromptRunner(None, TestLLMonPyPrompt("LLMonPy"), model_info)
            step_list.append(step)
        future_list = []
        result_dict = {}
        for step in step_list:
            future = step.get_thread_pool().submit(step.record_step)
            future_list.append(future)
        for future in concurrent.futures.as_completed(future_list):
            try:
                returned_step = future.result()
                result = returned_step.get_step_output()
                print(result.to_json())
            except Exception as e:
                print(str(e))  # exception was logged in record_step
                pass
        result_list = list(result_dict.values())
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(stack_trace)
        print(str(e))
    finally:
        llmonpy_stop()
        exit(0)
