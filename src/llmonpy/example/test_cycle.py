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

from llmonpy.llm_client import GPT4o, MISTRAL_LARGE, GEMINI_PRO, GEMINI_FLASH, ANTHROPIC_SONNET, MISTRAL_7B, ANTHROPIC_HAIKU, \
    MISTRAL_8X22B, ANTHROPIC_OPUS, MISTRAL_SMALL, filter_clients_that_didnt_start, GPT4omini
from llmonpy.llmon_pypeline import LLMonPypeline
from llmonpy.llmonpy_execute import run_step
from llmonpy.llmonpy_step import TraceLogRecorderInterface
from llmonpy.llmonpy_tournament import AdaptiveICLCycle
from llmonpy.system_startup import llmonpy_start, llmonpy_stop
from llmonpy.example.test_tourney import NameIterativeRefinementTournamentPrompt


class GenerateNameCycle(LLMonPypeline):
    def __init__(self):
        pass

    def execute_step(self, recorder: TraceLogRecorderInterface):
        first_round_client_list = filter_clients_that_didnt_start([GPT4o, MISTRAL_LARGE, GEMINI_PRO, ANTHROPIC_SONNET,
                                                                   ANTHROPIC_OPUS])
        client_list = filter_clients_that_didnt_start([GPT4o, GPT4omini, GEMINI_PRO, GEMINI_FLASH, ANTHROPIC_SONNET,
                                                       MISTRAL_7B, ANTHROPIC_HAIKU])
        judge_client_list = filter_clients_that_didnt_start([MISTRAL_SMALL, GEMINI_FLASH, MISTRAL_7B, GPT4omini,
                                                             ANTHROPIC_HAIKU])
        generator_prompt = NameIterativeRefinementTournamentPrompt()
        judgement_prompt = NameIterativeRefinementTournamentPrompt.JudgePrompt(generator_prompt)
        cycle = AdaptiveICLCycle(generator_prompt, client_list, [0.0, 0.75], judgement_prompt,
                                 judge_client_list, [0.0], 5, 3)
        result_list, _ = cycle.execute_step(recorder)
        for result in result_list:
            print("name:" + result.name)
        return result_list[0], recorder


if __name__ == "__main__":
    llmonpy_start()
    print("Running Test Cycle")
    try:
        step = GenerateNameCycle()
        result, recorder = run_step(step)
        print(result.to_json())
    except Exception as e:
        print(str(e))
    finally:
        llmonpy_stop()
        exit(0)


