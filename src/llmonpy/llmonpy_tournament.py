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
from llmonpy.llmonpy_execute import do_llmonpy_parallel_step, do_llmonpy_step
from llmonpy.llmonpy_prompt import LLMonPyPrompt, create_prompt_steps
from llmonpy.llmonpy_step import LLMonPyStep, LLMonPyStepOutput, TraceLogRecorderInterface, STEP_NAME_SEPARATOR, \
    DictLLMonPyStepOutput, JudgedOutput, STEP_TYPE_TOURNEY, STEP_TYPE_CYCLE, STEP_TYPE_JUDGE, STEP_TYPE_RANKER, \
    STEP_TYPE_JURY, STEP_TYPE_GENERATOR


class TournamentJudgePrompt(LLMonPyPrompt):
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


class TournamentGenerator(LLMonPypeline):
    def __init__(self, generation_prompt, generation_model_info_list):
        self.generation_prompt = generation_prompt
        self.generation_model_info_list = generation_model_info_list
        self.contestant_list = create_prompt_steps(generation_prompt, generation_model_info_list)

    def get_step_type(self) -> str:
        return STEP_TYPE_GENERATOR

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        generation_model_info_list = [model_info.to_dict() for model_info in self.generation_model_info_list]
        result = {"prompt_template": self.generation_prompt.get_prompt_text(),
                  "prompt_input_dict": self.generation_prompt.to_dict(),
                  "model_list": generation_model_info_list}
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        future_list = []
        output_list: [JudgedOutput] = []
        response_dict = {}
        for contestant in self.contestant_list:
            future, step_recorder = do_llmonpy_parallel_step(contestant, recorder)
            future_list.append(future)
        for future in concurrent.futures.as_completed(future_list):
            try:
                output, step_recorder = future.result()
                output_as_str = str(output)
                if output_as_str not in response_dict:
                    print("output received " + output_as_str)
                    response_dict[output_as_str] = output
                    judged_output = JudgedOutput(step_recorder.get_step_id(), output, step_recorder.get_model_info())
                    output_list.append(judged_output)
                else:
                    print("duplicate " + output_as_str)
                    recorder.log_message("duplicate " + output_as_str)
            except Exception as e:
                print(str(e))
                pass
        return output_list, recorder


class CompareOutputStep(LLMonPyStep):
    class LLMonPyOutput(LLMonPyPrompt.LLMonPyOutput):
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
        self.judge_list = create_prompt_steps(judgement_prompt, judgement_model_info_list)

    def get_step_type(self) -> str:
        return STEP_TYPE_JURY

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        judgement_model_info_list = [model_info.to_dict() for model_info in self.judgement_model_info_list]
        result = {"output_1": self.output_1.to_dict(), "output_2": self.output_2.to_dict(),
                  "judgement_prompt": self.judgement_prompt.get_prompt_text(),
                  "judgement_model_info_list": judgement_model_info_list}
        return result

    def execute_step(self, recorder):
        future_list = []
        contestant_1_victory_count = 0
        contestant_2_victory_count = 0
        for judge in self.judge_list:
            judge.get_prompt().set_contestants(self.output_1.step_output, self.output_2.step_output)
            future, step_recorder = do_llmonpy_parallel_step(judge, recorder)
            future_list.append(future)
        for future in concurrent.futures.as_completed(future_list):
            try:
                output, step_recorder = future.result()
                if output.winner == 1:
                    contestant_1_victory_count += 1
                else:
                    contestant_2_victory_count += 1
            except Exception as e:
                print(str(e))
                pass
        if contestant_1_victory_count > contestant_2_victory_count:
            self.winner = self.output_1
            self.dissent_count = contestant_2_victory_count
            print("winner 1")
        else:
            self.winner = self.output_2
            self.dissent_count = contestant_1_victory_count
            print("winner 2")
        result = CompareOutputStep.LLMonPyOutput(self.output_1.output_id, self.output_2.output_id, self.winner.output_id,
                                                 self.dissent_count)
        return result, recorder


class RankOutputStep(LLMonPyStep):
    def __init__(self, contestant_step_name, contestant_list: [JudgedOutput], judgement_prompt, judgement_model_info_list):
        self.contestant_step_name = contestant_step_name
        self.contestant_list = contestant_list
        self.judgement_prompt = judgement_prompt
        self.judgement_model_info_list = judgement_model_info_list

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
        tourney_result = recorder.create_tourney_result(number_of_judges, self.contestant_step_name)
        while start_index < (number_of_contestants - 1):
            for i in range(start_index + 1, number_of_contestants):
                contest_list.append(CompareOutputStep(self.contestant_list[start_index], self.contestant_list[i],
                                                      self.judgement_prompt, self.judgement_model_info_list))
            start_index += 1
        future_list = []
        print("number of contests " + str(len(contest_list)))
        for contest in contest_list:
            future, step_recorder = do_llmonpy_parallel_step(contest, recorder)
            future_list.append(future)
        for future in concurrent.futures.as_completed(future_list):
            try:
                contest_result, contest_recorder = future.result()
                tourney_result.add_contest_result(contest_result.output_1_id, contest_result.output_2_id,
                                                  contest_result.winner_id, contest_result.dissent_count)
                self.record_victory(contest_result.winner_id)
            except Exception as e:
                print(str(e))
                pass
        ordered_contestant_list = sorted(self.contestant_list, key=lambda x: x.victory_count, reverse=True)
        recorder.record_tourney_result(ordered_contestant_list, tourney_result)
        return ordered_contestant_list, recorder

    def record_victory(self, winner_id):
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
        generate_step = TournamentGenerator(self.generation_prompt, self.generation_model_info_list)
        output_list, _ = do_llmonpy_step(generate_step, recorder)
        rank_step = RankOutputStep(self.generation_prompt.get_short_step_name(), output_list, self.judgement_prompt,
                                   self.judgement_model_info_list)
        ordered_output_list, step_recorder = do_llmonpy_step(rank_step, recorder)
        return ordered_output_list, recorder

    def output_no_score(self, ordered_output_list: [JudgedOutput], recorder):
        ordered_output_list = [judged_output.step_output for judged_output in ordered_output_list]
        return ordered_output_list, recorder

    def winner_only(self, ordered_output_list: [JudgedOutput], recorder):
        result = ordered_output_list[0].step_output
        return result, recorder


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

    def execute_step(self, recorder: TraceLogRecorderInterface):
        tournament = LLMonPyTournament(self.generation_prompt, self.first_round_model_info_list,
                                       self.judgement_prompt, self.judgement_model_info_list)
        first_round_result_list, _ = do_llmonpy_step(tournament, recorder)
        self.update_example_list(first_round_result_list, recorder)
        for i in range(1, self.max_cycles):
            tournament = LLMonPyTournament(self.generation_prompt, self.generation_model_info_list,
                                           self.judgement_prompt, self.judgement_model_info_list)
            recorder.set_step_examples(self.generation_prompt_name, self.get_example_output_list())
            result_list, _ = do_llmonpy_step(tournament, recorder)
            if i  < self.max_cycles - 1:
                new_champion = self.update_example_list(result_list, recorder)
                if new_champion is False:
                    recorder.log_message("cycle done " + str(i) + " champion: " + str(self.example_list[0]))
                    result_list = self.get_example_output_list()
                    break
                else:
                    recorder.log_message("cycle " + str(i) + " champion: " + str(self.example_list[0]))
        return result_list, recorder

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
            rank_step = RankOutputStep(self.generation_prompt.get_short_step_name(), full_list, self.judgement_prompt,
                                       self.judgement_model_info_list)
            ordered_list, step_recorder = do_llmonpy_step(rank_step, recorder)
            self.example_list = ordered_list[0:self.number_of_examples]
            new_champion = self.example_list[0].output_id != current_champion.output_id
        return new_champion
