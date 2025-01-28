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
import json
import os
import uuid
import time

from llm_client import FIREWORKS_LLAMA3_1_405B, MINISTRAL_3B
from llmonpy.llm_client import GPT4o, MISTRAL_LARGE, GEMINI_PRO, GEMINI_FLASH, ANTHROPIC_SONNET, MISTRAL_7B, \
    ANTHROPIC_HAIKU, \
    MISTRAL_8X22B, ANTHROPIC_OPUS, MISTRAL_SMALL, filter_clients_that_didnt_start, GPT4omini, FIREWORKS_LLAMA3_1_8B, \
    FIREWORKS_MYTHOMAXL2_13B, TOMBU_LLAMA3_1_8B, TOMBU_DOLPHIN_QWEN2_72B, GROQ_LLAMA3_1_70B, GEMINI_FLASH_8B
from llmonpy.llmon_pypeline import LLMonPypeline
from llmonpy.llmonpy_step import TraceLogRecorderInterface, make_model_list, ModelTemp, LLMonPyStepOutput
from llmonpy.llmonpy_tournament import AdaptiveICLCycle
from llmonpy.system_startup import llmonpy_start, llmonpy_stop
from llmonpy.example.test_tourney import NameIterativeRefinementTournamentPrompt
from llmonpy.trace_log import trace_log_service


class GenerateNameCycle(LLMonPypeline):

    class LLMonPyOutput(LLMonPyStepOutput):
        def __init__(self, name=None):
            super().__init__()
            self.name = name

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            return result

    def __init__(self, first_round_info_list=None, aggregate_info_list=None, judge_client_info_list=None):
        self.first_round_info_list = first_round_info_list if first_round_info_list is not None \
            else make_model_list(ModelTemp([ANTHROPIC_SONNET, GEMINI_PRO, GPT4o, MISTRAL_LARGE, FIREWORKS_LLAMA3_1_405B], [0.0, 0.75]))
        self.aggregate_info_list = aggregate_info_list if aggregate_info_list is not None \
            else make_model_list(ModelTemp([GPT4omini, GEMINI_FLASH, FIREWORKS_LLAMA3_1_8B, ANTHROPIC_HAIKU], [0.0, 0.25, 0.50, 0.75]))
        self.judge_client_info_list = judge_client_info_list if judge_client_info_list is not None \
            else make_model_list(ModelTemp([GEMINI_FLASH, GEMINI_FLASH_8B, MISTRAL_7B, GPT4omini, ANTHROPIC_HAIKU],0.0))

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        return {}

    def execute_step(self, recorder: TraceLogRecorderInterface):
        generator_prompt = NameIterativeRefinementTournamentPrompt()
        judgement_prompt = NameIterativeRefinementTournamentPrompt.JudgePrompt(generator_prompt)
        cycle = AdaptiveICLCycle(generator_prompt, self.aggregate_info_list, judgement_prompt,
                                 self.judge_client_info_list, 5, 4, self.first_round_info_list).create_step(recorder)
        cycle.record_step()
        ordered_response_list = cycle.get_step_output().ordered_response_list
        for result in ordered_response_list:
            print("name:" + result.step_output.name)
        result = GenerateNameCycle.LLMonPyOutput(ordered_response_list[0].step_output.name)
        return result


def write_training_data(trace_id: str):
    tourney_result_list = trace_log_service().get_tourney_results_for_trace(trace_id)
    qbawa_list = []
    for tourney in tourney_result_list:
        tourney_qbawa = tourney.generate_qbawa()
        qbawa_list.extend(tourney_qbawa)
    qbawa_list = [qbawa.to_dict() for qbawa in qbawa_list]
    file_path = "data/training_data.json"
    with open(file_path, "w") as file:
        json.dump(qbawa_list, file, indent=4)

if __name__ == "__main__":
    llmonpy_start()
    print("Running Test Cycle")
    try:
        step = GenerateNameCycle().create_step(None)
        step.record_step()
        result = step.get_step_output()
        trace_id = step.get_recorder().get_trace_id()
        write_training_data(trace_id)
        print(result.to_json())
    except Exception as e:
        print(str(e))
    finally:
        llmonpy_stop()
        exit(0)


