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
import concurrent
import copy
import json
import uuid

from llmonpy.llmon_pypeline import LLMonPypeline
from llmonpy.llmonpy_prompt import LLMonPyPrompt, create_prompt_steps, JudgePrompt, LLMonPyPromptRunner
from llmonpy.llmonpy_step import LLMonPyStep, LLMonPyStepOutput, TraceLogRecorderInterface, STEP_NAME_SEPARATOR, \
    DictLLMonPyStepOutput, JudgedOutput, STEP_TYPE_TOURNEY, STEP_TYPE_CYCLE, STEP_TYPE_JUDGE, STEP_TYPE_RANKER, \
    STEP_TYPE_JURY, STEP_TYPE_GENERATOR


class TournamentJudgePrompt(JudgePrompt):
    class LLMonPyOutput(LLMonPyPrompt.LLMonPyOutput):
        def __init__(self, winner: int):
            self.winner: int = winner

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            return result

        @staticmethod
        def from_dict(dictionary):
            result = TournamentJudgePrompt.LLMonPyOutput(**dictionary)
            return result

    def __init__(self, step_being_judged):
        self.step_being_judged = step_being_judged
        self.name_of_step_being_judged = step_being_judged.get_step_name()

    def get_step_name(self):
        result = (self.name_of_step_being_judged + STEP_NAME_SEPARATOR + self.__class__.__module__ + "."
                  + self.__class__.__name__)
        return result

    def get_step_type(self) -> str:
        return STEP_TYPE_JUDGE

    def set_contestants(self, contestant_1, contestant_2):
        raise NotImplementedError()

    def output_from_dict(self, output_dict):
        result = TournamentJudgePrompt.LLMonPyOutput.from_dict(output_dict)
        return result

    def to_dict(self):
        result = copy.deepcopy(vars(self))
        result["step_being_judged"] = self.step_being_judged.to_dict()
        return result


class TournamentResponseGenerator(LLMonPypeline):

    class LLMonPyOutput(LLMonPyStepOutput):
        def __init__(self, response_list: [JudgedOutput]):
            self.response_list: [JudgedOutput] = response_list

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            for i in range(0, len(self.response_list)):
                result["response_list"][i] = self.response_list[i].to_dict()
            return result

    def __init__(self, generation_prompt, generation_model_info_list):
        self.generation_prompt = generation_prompt
        self.generation_model_info_list = generation_model_info_list
        self.contestant_list = None
        self.output_list: [JudgedOutput] = []
        self.response_dict = {}

    def get_step_type(self) -> str:
        return STEP_TYPE_GENERATOR

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        generation_model_info_list = [model_info.to_dict() for model_info in self.generation_model_info_list]
        result = {"prompt_template": self.generation_prompt.get_prompt_text(),
                  "prompt_input_dict": self.generation_prompt.to_dict(),
                  "model_list": generation_model_info_list}
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        self.contestant_list = create_prompt_steps(recorder, self.generation_prompt, self.generation_model_info_list)
        self.run_parallel_steps(self.contestant_list, handle_result_function=self.record_output)
        result = TournamentResponseGenerator.LLMonPyOutput(self.output_list)
        return result

    def record_output(self, step):
        output = step.get_step_output()
        output_as_str = str(output)
        if output_as_str not in self.response_dict:
            print("output received " + output_as_str)
            self.response_dict[output_as_str] = output
            judged_output = JudgedOutput(step.get_step_id(), output, step.get_model_info())
            self.output_list.append(judged_output)
        else:
            print("duplicate " + output_as_str)


class CompareOutputStep(LLMonPypeline):
    class LLMonPyOutput(LLMonPyStepOutput):
        def __init__(self, output_1_id: str, output_2_id: str, winner_id: str, dissent_count: int = 0):
            self.output_1_id: str = output_1_id
            self.output_2_id: str = output_2_id
            self.winner_id: str = winner_id
            self.dissent_count: int = dissent_count

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            return result

    def __init__(self, output_1, output_2, judgement_prompt, judgement_model_info_list):
        self.output_1 = output_1
        self.output_2 = output_2
        self.winner = None
        self.dissent_count = 0
        self.judgement_prompt = judgement_prompt
        self.judgement_model_info_list = judgement_model_info_list
        self.judge_list = None
        self.contestant_1_victory_count = 0
        self.contestant_2_victory_count = 0

    def get_step_type(self) -> str:
        return STEP_TYPE_JURY

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        judgement_model_info_list = [model_info.to_dict() for model_info in self.judgement_model_info_list]
        result = {"output_1": self.output_1.to_dict(), "output_2": self.output_2.to_dict(),
                  "judgement_prompt": self.judgement_prompt.get_prompt_text(),
                  "judgement_model_info_list": judgement_model_info_list}
        return result

    def execute_step(self, recorder):
        self.judge_list = create_prompt_steps(recorder, self.judgement_prompt, self.judgement_model_info_list)
        for judge in self.judge_list:
            judge.get_prompt().set_contestants(self.output_1.step_output, self.output_2.step_output)
        self.run_parallel_steps(self.judge_list, handle_result_function=self.record_victory)
        if self.contestant_1_victory_count > self.contestant_2_victory_count:
            self.winner = self.output_1
            self.dissent_count = self.contestant_2_victory_count
        else:
            self.winner = self.output_2
            self.dissent_count = self.contestant_1_victory_count
        result = CompareOutputStep.LLMonPyOutput(self.output_1.output_id, self.output_2.output_id, self.winner.output_id,
                                                 self.dissent_count)
        return result

    def record_victory(self, step):
        output = step.get_step_output()
        if output.winner == 1:
            self.contestant_1_victory_count += 1
        else:
            self.contestant_2_victory_count += 1


class OrderedStepOutputList(LLMonPyStepOutput):
        def __init__(self, ordered_response_list: [JudgedOutput]):
            self.ordered_response_list: [JudgedOutput] = ordered_response_list

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            for i in range(0, len(self.ordered_response_list)):
                result["ordered_response_list"][i] = self.ordered_response_list[i].to_dict()
            return result


class RankOutputStep(LLMonPypeline):
    def __init__(self, prompt, contestant_list: [JudgedOutput], judgement_prompt, judgement_model_info_list):
        self.request_text = LLMonPyPromptRunner.render_prompt(prompt)
        self.contestant_step_name = prompt.get_short_step_name()
        self.contestant_list = contestant_list
        self.judgement_prompt = judgement_prompt
        self.judgement_model_info_list = judgement_model_info_list
        self.tourney_result = None

    def get_step_type(self) -> str:
        return STEP_TYPE_RANKER

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        contestant_dict_list = [contestant.to_dict() for contestant in self.contestant_list]
        judgement_model_info_list = [model_info.to_dict() for model_info in self.judgement_model_info_list]
        result = {"contestant_step_name": self.contestant_step_name,  "contestant_list": contestant_dict_list,
                  "judgement_prompt": self.judgement_prompt.get_prompt_text(),
                  "judgement_model_info_list": judgement_model_info_list}
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        start_index = 0
        contest_list = []
        number_of_contestants = len(self.contestant_list)
        number_of_judges = len(self.judgement_model_info_list)
        self.tourney_result = recorder.create_tourney_result(self.request_text, number_of_judges, self.contestant_step_name)
        while start_index < (number_of_contestants - 1):
            for i in range(start_index + 1, number_of_contestants):
                contest_list.append(CompareOutputStep(self.contestant_list[start_index], self.contestant_list[i],
                                                      self.judgement_prompt, self.judgement_model_info_list).create_step(recorder))
            start_index += 1
        print("number of contests " + str(len(contest_list)))
        self.run_parallel_steps(contest_list, handle_result_function=self.record_victory)
        ordered_contestant_list = sorted(self.contestant_list, key=lambda x: x.victory_count, reverse=True)
        recorder.record_tourney_result(ordered_contestant_list, self.tourney_result)
        result = OrderedStepOutputList(ordered_contestant_list)
        return result

    def record_victory(self, step):
        contest_result = step.get_step_output()
        self. tourney_result.add_contest_result(step.get_step_id(),contest_result.output_1_id, contest_result.output_2_id,
                                          contest_result.winner_id, contest_result.dissent_count)
        winner_id = contest_result.winner_id
        for contestant in self.contestant_list:
            if contestant.output_id == winner_id:
                contestant.victory_count += 1
                break


class LLMonPyTournament(LLMonPypeline):

    def __init__(self, generation_prompt, generation_model_info_list, judgement_prompt,
                 judgement_model_info_list):
        self.generation_prompt = generation_prompt
        self.generation_model_info_list = generation_model_info_list
        self.judgement_prompt = judgement_prompt
        self.judgement_model_info_list = judgement_model_info_list

    def get_step_type(self) -> str:
        return STEP_TYPE_TOURNEY

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        generation_model_info_list = [model_info.to_dict() for model_info in self.generation_model_info_list]
        judgement_model_info_list = [model_info.to_dict() for model_info in self.judgement_model_info_list]
        result = {"generation_prompt": self.generation_prompt.get_prompt_text(),
                  "generation_model_info_list": generation_model_info_list,
                  "judgement_prompt": self.judgement_prompt.get_prompt_text(),
                  "judgement_model_info_list": judgement_model_info_list}
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        output_list: [JudgedOutput] = []
        generate_step = TournamentResponseGenerator(self.generation_prompt, self.generation_model_info_list).create_step(recorder)
        generate_step.record_step()
        response_list = generate_step.get_step_output().response_list
        rank_step = RankOutputStep(self.generation_prompt, response_list, self.judgement_prompt,
                                   self.judgement_model_info_list).create_step(recorder)
        rank_step.record_step()
        result = OrderedStepOutputList(rank_step.get_step_output().ordered_response_list)
        return result


class AdaptiveICLCycle(LLMonPypeline):

    def __init__(self, generation_prompt, generation_model_info_list, judgement_prompt,
                 judgement_model_info_list, number_of_examples: int = 1, max_cycles: int = 4,
                 first_round_model_info_list=None):
        self.generation_prompt = generation_prompt
        self.generation_prompt_name = self.generation_prompt.get_step_name()
        self.generation_model_info_list = generation_model_info_list
        self.judgement_prompt = judgement_prompt
        self.judgement_model_info_list = judgement_model_info_list
        self.first_round_model_info_list = first_round_model_info_list if first_round_model_info_list is not None else generation_model_info_list
        self.number_of_examples = number_of_examples
        self.max_cycles = max_cycles
        self.example_list: [JudgedOutput] = []

    def get_step_type(self) -> str:
        return STEP_TYPE_CYCLE

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        generation_model_info_list = [model_info.to_dict() for model_info in self.generation_model_info_list]
        judgement_model_info_list = [model_info.to_dict() for model_info in self.judgement_model_info_list]
        first_round_model_info_list = [model_info.to_dict() for model_info in self.first_round_model_info_list]
        result = {"generation_prompt": self.generation_prompt.get_prompt_text(),
                  "generation_model_info_list": generation_model_info_list,
                  "judgement_prompt": self.judgement_prompt.get_prompt_text(),
                  "judgement_model_info_list": judgement_model_info_list,
                  "first_round_model_info_list": first_round_model_info_list,
                  "number_of_examples": self.number_of_examples, "max_cycles": self.max_cycles}
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        tournament = LLMonPyTournament(self.generation_prompt, self.first_round_model_info_list,
                                       self.judgement_prompt, self.judgement_model_info_list).create_step(recorder)
        tournament.record_step()
        first_round_result_list = tournament.get_step_output().ordered_response_list
        self.update_example_list(first_round_result_list, recorder)
        for i in range(1, self.max_cycles):
            tournament = LLMonPyTournament(self.generation_prompt, self.generation_model_info_list,
                                           self.judgement_prompt, self.judgement_model_info_list).create_step(recorder)
            recorder.set_step_examples(self.generation_prompt_name, self.get_example_output_list())
            tournament.record_step()
            result_list = tournament.get_step_output().ordered_response_list
            new_champion = self.update_example_list(result_list, recorder)
            if new_champion is False:
                recorder.log_message("cycle done " + str(i) + " champion: " + str(self.example_list[0]))
                break
            else:
                recorder.log_message("cycle " + str(i) + " champion: " + str(self.example_list[0]))
        result = OrderedStepOutputList(self.example_list)
        return result

    def get_example_output_list(self):
        result = [judged_output.step_output for judged_output in self.example_list]
        result = result[::-1]
        return result

    def update_example_list(self, result_list: [JudgedOutput], recorder):
        new_champion = False
        best_list = result_list[0:self.number_of_examples]
        if len(self.example_list) == 0:
            self.example_list = best_list
            new_champion = True
        else:
            current_champion = self.example_list[0]
            full_list = best_list + self.example_list
            for judged_output in full_list:
                judged_output.reset_victory_count()
            rank_step = RankOutputStep(self.generation_prompt, full_list, self.judgement_prompt,
                                       self.judgement_model_info_list).create_step(recorder)
            rank_step.record_step()
            ordered_response_list = rank_step.get_step_output().ordered_response_list
            self.example_list = ordered_response_list[0:self.number_of_examples]
            new_champion = self.example_list[0].output_id != current_champion.output_id
        return new_champion
