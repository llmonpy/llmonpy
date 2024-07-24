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
import os
import uuid
import time

from llmonpy.llm_client import GPT4o, MISTRAL_LARGE, GEMINI_PRO, GEMINI_FLASH, ANTHROPIC_SONNET, MISTRAL_7B, \
    ANTHROPIC_HAIKU, \
    MISTRAL_8X22B, ANTHROPIC_OPUS, MISTRAL_SMALL, filter_clients_that_didnt_start, GPT4omini, FIREWORKS_QWEN2_72B, \
    FIREWORKS_MYTHOMAXL2_13B, FIREWORKS_LLAMA3_1_8B
from llmonpy.llmon_pypeline import LLMonPypeline
from llmonpy.llmonpy_execute import run_step
from llmonpy.llmonpy_step import TraceLogRecorderInterface, make_model_list, ModelTemp
from llmonpy.llmonpy_tournament import AdaptiveICLCycle
from llmonpy.system_startup import llmonpy_start, llmonpy_stop
from llmonpy.example.test_tourney import NameIterativeRefinementTournamentPrompt
from llmonpy.llmonpy_gar import GenerateAggregateRankStep


class GenerateNameGar(LLMonPypeline):
    def __init__(self):
        pass

    def execute_step(self, recorder: TraceLogRecorderInterface):
        generate_list = [GPT4o, GEMINI_PRO, ANTHROPIC_SONNET, GEMINI_FLASH, GPT4omini, FIREWORKS_QWEN2_72B]
        generate_info_list = make_model_list(ModelTemp(generate_list, [0.0, 0.75]))
        aggregate_list = [GPT4omini, GEMINI_FLASH, ANTHROPIC_SONNET, FIREWORKS_MYTHOMAXL2_13B, ANTHROPIC_HAIKU]
        aggregate_info_list = make_model_list(ModelTemp(aggregate_list, [0.0, 0.25, 0.5, 0.75]))
        judge_client_info_list = make_model_list(ModelTemp([FIREWORKS_MYTHOMAXL2_13B, GEMINI_FLASH, FIREWORKS_LLAMA3_1_8B, GPT4omini,
                                                             ANTHROPIC_HAIKU],0.0))
        generator_prompt = NameIterativeRefinementTournamentPrompt()
        judgement_prompt = NameIterativeRefinementTournamentPrompt.JudgePrompt(generator_prompt)
        cycle = GenerateAggregateRankStep(generator_prompt, generate_info_list, aggregate_info_list,4,
                                           judgement_prompt, judge_client_info_list)
        result_list, _ = cycle.execute_step(recorder)
        for result in result_list:
            print("name:" + result.step_output.name)
        return result_list[0].step_output, recorder


if __name__ == "__main__":
    llmonpy_start()
    print("Running Test Gar")
    try:
        step = GenerateNameGar()
        result, recorder = run_step(step)
        print(result.to_json())
    except Exception as e:
        print(str(e))
    finally:
        llmonpy_stop()
        exit(0)


