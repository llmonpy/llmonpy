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

from llmon_pypeline import LLMonPypeline
from llmonpy_execute import do_llmonpy_parallel_step, do_llmonpy_step
from prompt import LLMonPyPrompt
from llmonpy_step import LLMonPyStep, LLMonPyStepOutput, TraceLogRecorderInterface, STEP_NAME_SEPARATOR, \
    DictLLMonPyStepOutput, JudgedOutput, STEP_TYPE_TOURNEY, STEP_TYPE_CYCLE, STEP_TYPE_JUDGE


def judge_output(contestant_list, judge_list, thread_pool, recorder):
    start_index = 0
    contest_list = []
    number_of_contestants = len(contestant_list)
    judge_step_name = judge_list[0].get_step_name()
    tourney_result = recorder.create_tourney_result(len(judge_list), judge_step_name)
    while start_index < (number_of_contestants - 1):
        for i in range(start_index + 1, number_of_contestants):
            contest_list.append(JudgedContest(contestant_list[start_index], contestant_list[i]))
        start_index += 1
    future_list = []
    print("number of contests " + str(len(contest_list)))
    for contest in contest_list:
        future = thread_pool.submit(contest.determine_winner, recorder, judge_list)
        future_list.append(future)
    for future in concurrent.futures.as_completed(future_list):
        try:
            result = future.result()
            tourney_result.add_contest_result(result.output_1.output_id, result.output_2.output_id,
                                              result.winner.output_id, result.dissent_count)
            result.winner.victory_count += 1
        except Exception as e:
            print(str(e))
            pass
    ordered_contestant_list = sorted(contestant_list, key=lambda x: x.victory_count, reverse=True)
    recorder.record_tourney_result(ordered_contestant_list, tourney_result)
    return ordered_contestant_list


class TournamentJudgePrompt(LLMonPyPrompt):
    class LLMonPyOutput(LLMonPyPrompt.LLMonPyOutput):
        def __init__(self, winner):
            self.winner = winner

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            return result

        @staticmethod
        def from_dict(dictionary):
            result = TournamentJudgePrompt.LLMonPyOutput(**dictionary)
            return result

    def __init__(self, name_of_step_being_judged):
        self.name_of_step_being_judged = name_of_step_being_judged

    def get_step_name(self):
        result = (self.name_of_step_being_judged + STEP_NAME_SEPARATOR + self.__class__.__module__ + "."
                  + self.__class__.__name__)
        return result

    def get_step_type(self) -> str:
        return STEP_TYPE_JUDGE

    def set_values(self, candidate_1, candidate_2):
        raise NotImplementedError()

    def output_from_dict(self, output_dict):
        result = TournamentJudgePrompt.LLMonPyOutput.from_dict(output_dict)
        return result


class JudgedContest:
    def __init__(self, output_1, output_2):
        self.output_1 = output_1
        self.output_2 = output_2
        self.winner = None
        self.dissent_count = 0

    def determine_winner(self, recorder, judge_list:[TournamentJudgePrompt]):
        future_list = []
        contestant_1_victory_count = 0
        contestant_2_victory_count = 0
        for judge in judge_list:
            judge.get_prompt().set_values(self.output_1.step_output, self.output_2.step_output)
            future, step_recorder = do_llmonpy_parallel_step(judge, recorder)
            future_list.append(future)
        for future in concurrent.futures.as_completed(future_list):
            try:
                output, step = future.result()
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
        return self


class LLMonPyTournament(LLMonPypeline):
    def __init__(self, contestant_list:[LLMonPyStep], judge_list:[LLMonPyStep], example_list=None):
        self.contestant_list = contestant_list
        self.judge_list = judge_list

    def get_step_type(self) -> str:
        return STEP_TYPE_TOURNEY

    def execute_step(self, recorder: TraceLogRecorderInterface):
        future_list = []
        output_list:[JudgedOutput] = []
        response_dict = {}
        for contestant in self.contestant_list:
            future, step_recorder = do_llmonpy_parallel_step(contestant, recorder)
            future_list.append(future)
        for future in concurrent.futures.as_completed(future_list):
            try:
                output, _ = future.result()
                output_as_str = str(output)
                if output_as_str not in response_dict:
                    print("output received " + output_as_str)
                    response_dict[output_as_str] = output
                    judged_output = JudgedOutput(output)
                    output_list.append(judged_output)
                else:
                    print("duplicate " + output_as_str)
                    recorder.log_message("duplicate " + output_as_str)
            except Exception as e:
                print(str(e))
                pass
        ordered_output_list = judge_output(output_list, self.judge_list, self.get_thread_pool(), recorder)
        return ordered_output_list, recorder

    def output_no_score(self, ordered_output_list:[JudgedOutput], recorder):
        ordered_output_list = [judged_output.step_output for judged_output in ordered_output_list]
        return ordered_output_list, recorder

    def winner_only(self, ordered_output_list:[JudgedOutput], recorder):
        result = ordered_output_list[0].step_output
        return result, recorder


class RefinementCycle(LLMonPypeline):
    def __init__(self, generation_prompt_name, contestant_list:[LLMonPyStep], judge_list:[LLMonPyStep],
                 first_round_contestant_list=None, number_of_examples:int = 1, max_cycles:int = 4):
        self.generation_prompt_name = generation_prompt_name
        self.contestant_list = contestant_list
        self.first_round_contestant_list = first_round_contestant_list if first_round_contestant_list is not None else contestant_list
        self.judge_list = judge_list
        self.number_of_examples = number_of_examples
        self.max_cycles = max_cycles
        self.example_list:[LLMonPyPrompt.LLMonPyOutput] = []

    def get_step_type(self) -> str:
        return STEP_TYPE_CYCLE

    def execute_step(self, recorder: TraceLogRecorderInterface):
        tournament = LLMonPyTournament(self.first_round_contestant_list, self.judge_list)
        first_round_result_list, _ = tournament.output_no_score(*do_llmonpy_step(tournament, recorder))
        self.update_example_list(first_round_result_list, recorder)
        for i in range(1, self.max_cycles):
            tournament = LLMonPyTournament(self.contestant_list, self.judge_list)
            recorder.set_step_examples(self.generation_prompt_name, self.example_list)
            result_list, _ = tournament.output_no_score(*do_llmonpy_step(tournament, recorder))
            new_champion = self.update_example_list(result_list, recorder)
            if new_champion is False:
                recorder.log_message("cycle done " + str(i) + " champion: " + str(self.example_list[0]))
                break
            else:
                recorder.log_message("cycle " + str(i) + " champion: " + str(self.example_list[0]))
        return self.example_list, recorder

    def update_example_list(self, result_list:[JudgedOutput], recorder):
        new_champion = False
        best_list = result_list[0:self.number_of_examples]
        if len(self.example_list) == 0:
            self.example_list = best_list
            new_champion = True
        else:
            current_champion = self.example_list[0]
            full_list = best_list + self.example_list
            full_list = [JudgedOutput(step_output) for step_output in full_list]
            ordered_list = judge_output(full_list, self.judge_list, self.get_thread_pool(), recorder)
            self.example_list = ordered_list[0:self.number_of_examples]
            self.example_list = [judged_output.step_output for judged_output in self.example_list]
            new_champion = str(self.example_list[0]) != str(current_champion)
        return new_champion

