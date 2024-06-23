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

from llm_client import GPT4o, MISTRAL_LARGE, GEMINI_PRO, GEMINI_FLASH, ANTHROPIC_SONNET, MISTRAL_7B, ANTHROPIC_HAIKU, \
    MISTRAL_8X22B, ANTHROPIC_OPUS, ALL_CLIENT_LIST
from llmon_pypeline import LLMonPypeline
from llmonpy_step import TraceLogRecorderInterface
from llmonpy_tournament import ChampionCycle
from prompt import create_prompt_steps
from system_startup import system_startup, system_stop
from test_tourney import NameIterativeRefinementTournamentPrompt
from trace_log import trace_log_service


class GenerateNameCycle(LLMonPypeline):
    def __init__(self):
        pass

    def execute_step(self, recorder: TraceLogRecorderInterface):
        first_round_client_list = [GPT4o, MISTRAL_LARGE, GEMINI_PRO, ANTHROPIC_SONNET, ANTHROPIC_OPUS]
        client_list = [GPT4o, MISTRAL_LARGE, GEMINI_PRO, GEMINI_FLASH, ANTHROPIC_SONNET, MISTRAL_7B, ANTHROPIC_HAIKU]
        judge_client_list = [MISTRAL_LARGE, GEMINI_FLASH, MISTRAL_7B, MISTRAL_8X22B, ANTHROPIC_HAIKU]

        first_round_generators = create_prompt_steps(NameIterativeRefinementTournamentPrompt(), first_round_client_list, [0.0])
        generators = create_prompt_steps(NameIterativeRefinementTournamentPrompt(), client_list, [0.0])
        judge_list = create_prompt_steps(NameIterativeRefinementTournamentPrompt.JudgePrompt(), judge_client_list)
        tournament = ChampionCycle(generators, judge_list, first_round_generators, 3, 3 )
        result_list, _ = tournament.execute_step(recorder)
        for result in result_list:
            print("name:" + result.name)
        return result_list[0], recorder


if __name__ == "__main__":
    system_startup()
    print("Running Test Cycle")
    step = GenerateNameCycle()
    trace_id = str(uuid.uuid4())
    recorder = trace_log_service().create_root_recorder(trace_id, trace_id, None, step)
    result, _ = step.execute_step(recorder)
    recorder.finish_child_step(result)
    print(result.to_json())
    system_stop()
    exit(0)


