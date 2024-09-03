import concurrent
import traceback

from llmonpy.llm_client import FIREWORKS_LLAMA3_1_8B, MISTRAL_7B, GPT4omini, GEMINI_FLASH, TOMBU_LLAMA3_1_8B
from llmonpy.llmonpy_prompt import LLMonPyPromptRunner
from llmonpy.llmonpy_step import LlmModelInfo
from llmonpy.system_startup import llmonpy_start, llmonpy_stop
from llmonpy.example.test_prompts import TestLLMonPyPrompt

if __name__ == "__main__":
    llmonpy_start()
    try:
        print("Running Test Rate Limiter")
        step_list = []
        for i in range(200):
            model_info = LlmModelInfo(TOMBU_LLAMA3_1_8B.model_name)
            step = LLMonPyPromptRunner(None, TestLLMonPyPrompt("LLMonPy"), model_info)
            step_list.append(step)
        future_list = []
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
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(stack_trace)
        print(str(e))
    finally:
        llmonpy_stop()
        exit(0)
