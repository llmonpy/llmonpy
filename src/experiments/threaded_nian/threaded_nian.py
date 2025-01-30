import sys
import traceback

from llm_client import GPT4omini, GEMINI_FLASH, ANTHROPIC_SONNET, ANTHROPIC_HAIKU, GEMINI_FLASH_8B, GPT4o, \
    MISTRAL_LARGE, MISTRAL_NEMO_12B, FIREWORKS_DEEPSEEK_V3, GEMINI_PRO, GEMINI_FLASH_2
from llmonpy.system_startup import llmonpy_start, llmonpy_stop
from llmonpy.llmonpy_prompt import LLMonPySimplePrompt, LLMonPyPromptRunner
from llmonpy.llmonpy_step import LlmModelInfo

class ThreadedNianTest:
    def __init__(self, model_name):
        self.model_name = model_name
        self.response_string = None

    def run(self, file_path):
        print(f"Running model:{self.model_name}")
        simple_prompt = LLMonPySimplePrompt.from_file(file_path, "threaded_nian")
        model_info = LlmModelInfo(self.model_name)
        step = LLMonPyPromptRunner(None, simple_prompt, model_info)
        step.record_step()
        result = step.get_step_output()
        self.response_string = result.response_string
        self.report()

    def report(self):
        print(f"Model:{self.model_name} Response: {self.response_string}")

models_to_test = [
    #ThreadedNianTest(MISTRAL_NEMO_12B.model_name),
    # ThreadedNianTest(MISTRAL_LARGE.model_name),
    #ThreadedNianTest(FIREWORKS_DEEPSEEK_V3.model_name),
    #ThreadedNianTest(GPT4omini.model_name),
    # ThreadedNianTest(GPT4o.model_name),
    #ThreadedNianTest(GEMINI_FLASH_2.model_name),
    #ThreadedNianTest(GEMINI_FLASH.model_name),
    #ThreadedNianTest(GEMINI_FLASH_8B.model_name),
    #ThreadedNianTest(ANTHROPIC_SONNET.model_name),
    #ThreadedNianTest(ANTHROPIC_HAIKU.model_name),

]

if __name__ == "__main__":
    llmonpy_start()
    try:
        prompt_path="src/experiments/threaded_nian/prompts/120k_spread_q5.txt"
        for model in models_to_test:
            model.run(prompt_path)
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(stack_trace)
        print("Error:", str(e))
    finally:
        llmonpy_stop()
        print("Results:")
        for model in models_to_test:
            model.report()
        exit(0)
