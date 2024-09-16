import traceback

from llmon_pypeline import LLMonPypeline
from llmonpy.llm_client import GPT4omini, GEMINI_FLASH, FIREWORKS_MYTHOMAXL2_13B, FIREWORKS_LLAMA3_1_8B, \
    ANTHROPIC_HAIKU, MISTRAL_7B, GPT4o, ANTHROPIC_SONNET, GEMINI_PRO, FIREWORKS_LLAMA3_1_405B, MISTRAL_NEMO_12B, GPT3_5, \
    MISTRAL_LARGE, AI21_JAMBA_1_5_MINI
from llmonpy_prompt import create_prompt_steps
from llmonpy_step import TraceLogRecorderInterface, make_model_list, ModelTemp
from system_startup import llmonpy_start, llmonpy_stop
from test_tourney import NameIterativeRefinementTournamentPrompt

TEST_MODEL_LIST =[ANTHROPIC_HAIKU, MISTRAL_NEMO_12B, GEMINI_FLASH, FIREWORKS_LLAMA3_1_8B, GPT4omini]


class TestRatellmiterPypeLine(LLMonPypeline):
    def __init__(self, test_prompt, model_info_list):
        self.test_prompt = test_prompt
        self.model_info_list = model_info_list

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        model_info_list = [model_info.to_dict() for model_info in self.model_info_list]
        result = {"model_list": model_info_list}
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        step_list = []
        for i in range(0, 100):
            set_step_list = create_prompt_steps(recorder, self.test_prompt, self.model_info_list)
            step_list.extend(set_step_list)
        self.run_parallel_steps(step_list)
        response_list = [step.get_step_output() for step in step_list]
        return response_list


if __name__ == "__main__":
    llmonpy_start()
    try:
        print("Running TestRatellmiterPypeLine")
        model_info_list = make_model_list(ModelTemp(TEST_MODEL_LIST, [0.0]))
        step = TestRatellmiterPypeLine(NameIterativeRefinementTournamentPrompt(), model_info_list).create_step(None)
        step.record_step()
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(stack_trace)
        print(str(e))
    finally:
        llmonpy_stop()
        exit(0)
