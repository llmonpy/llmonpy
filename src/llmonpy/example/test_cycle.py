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
import uuid
import time

from llmonpy.llm_client import GPT4o, MISTRAL_LARGE, GEMINI_PRO, GEMINI_FLASH, ANTHROPIC_SONNET, MISTRAL_7B, \
    ANTHROPIC_HAIKU, \
    MISTRAL_8X22B, ANTHROPIC_OPUS, MISTRAL_SMALL, filter_clients_that_didnt_start, GPT4omini, FIREWORKS_LLAMA3_1_8B, \
    FIREWORKS_QWEN2_72B, FIREWORKS_MYTHOMAXL2_13B
from llmonpy.llmon_pypeline import LLMonPypeline
from llmonpy.llmonpy_execute import run_step
from llmonpy.llmonpy_step import TraceLogRecorderInterface, make_model_list, ModelTemp, LLMonPyStepOutput
from llmonpy.llmonpy_tournament import AdaptiveICLCycle
from llmonpy.system_startup import llmonpy_start, llmonpy_stop
from llmonpy.example.test_tourney import NameIterativeRefinementTournamentPrompt


class GenerateNameCycle(LLMonPypeline):

    class LLMonPyOutput(LLMonPyStepOutput):
        def __init__(self, name=None):
            super().__init__()
            self.name = name

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            return result

    def __init__(self):
        pass

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        return {}

    def execute_step(self, recorder: TraceLogRecorderInterface):
        first_round_list = [ANTHROPIC_SONNET, GEMINI_FLASH, GPT4omini, FIREWORKS_QWEN2_72B]
        first_round_info_list = make_model_list(ModelTemp(first_round_list, [0.0, 0.75]))
        aggregate_list = [GPT4omini, GEMINI_FLASH, FIREWORKS_MYTHOMAXL2_13B, ANTHROPIC_HAIKU, FIREWORKS_QWEN2_72B]
        aggregate_info_list = make_model_list(ModelTemp(aggregate_list, [0.0, 0.75]))
        judge_client_info_list = make_model_list(ModelTemp([FIREWORKS_LLAMA3_1_8B, GEMINI_FLASH,
                                                            FIREWORKS_MYTHOMAXL2_13B, GPT4omini,
                                                             ANTHROPIC_HAIKU],0.0))
        generator_prompt = NameIterativeRefinementTournamentPrompt()
        judgement_prompt = NameIterativeRefinementTournamentPrompt.JudgePrompt(generator_prompt)
        cycle = AdaptiveICLCycle(generator_prompt, aggregate_info_list, judgement_prompt,
                                 judge_client_info_list, 5, 2, first_round_info_list).create_step(recorder)
        cycle.record_step()
        ordered_response_list = cycle.get_step_output().ordered_response_list
        for result in ordered_response_list:
            print("name:" + result.step_output.name)
        result = GenerateNameCycle.LLMonPyOutput(ordered_response_list[0].step_output.name)
        return result


if __name__ == "__main__":
    llmonpy_start()
    print("Running Test Cycle")
    try:
        step = GenerateNameCycle().create_step(None)
        step.record_step()
        result = step.get_step_output()
        print(result.to_json())
    except Exception as e:
        print(str(e))
    finally:
        llmonpy_stop()
        exit(0)


