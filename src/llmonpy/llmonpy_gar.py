from llmonpy.llmon_pypeline import LLMonPypeline
from llmonpy.llmonpy_step import STEP_TYPE_GAR, TraceLogRecorderInterface, JudgedOutput
from llmonpy.llmonpy_tournament import TournamentResponseGenerator, RankOutputStep, OrderedStepOutputList


class GenerateAggregateRankStep(LLMonPypeline):
    def __init__(self, generation_prompt, generation_model_info_list, aggregation_model_info_list,
                 repeat_aggregation_layer: int = 2,
                 judgement_prompt = None, judgement_model_info_list = None ):
        self.generation_prompt = generation_prompt
        self.generation_model_info_list = generation_model_info_list
        self.aggregation_model_info_list = aggregation_model_info_list
        self.judgement_prompt = judgement_prompt
        self.judgement_model_info_list = judgement_model_info_list
        self.repeat_aggregation_layer = repeat_aggregation_layer

    def get_step_type(self) -> str:
        return STEP_TYPE_GAR

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        generation_model_info_list = [model_info.to_dict() for model_info in self.generation_model_info_list]
        aggregation_model_info_list = [model_info.to_dict() for model_info in self.aggregation_model_info_list]
        judgement_model_info_list = [model_info.to_dict() for model_info in self.judgement_model_info_list]
        result = {"generation_prompt": self.generation_prompt.get_prompt_text(),
                  "generation_model_info_list": generation_model_info_list,
                  "aggregation_model_info_list": aggregation_model_info_list,
                  "judgement_prompt": self.judgement_prompt.get_prompt_text(),
                  "judgement_model_info_list": judgement_model_info_list}
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        judged_output_list: [JudgedOutput] = []
        generate_step = TournamentResponseGenerator(self.generation_prompt, self.generation_model_info_list).create_step(recorder)
        generate_step.record_step()
        judged_output_list = generate_step.get_step_output().response_list
        step_output_list = [judged_output.step_output for judged_output in judged_output_list]
        recorder.set_step_examples(self.generation_prompt.get_step_name(), step_output_list)
        for i in range(0, self.repeat_aggregation_layer):
            generate_step = TournamentResponseGenerator(self.generation_prompt, self.aggregation_model_info_list).create_step(recorder)
            generate_step.record_step()
            judged_output_list = generate_step.get_step_output().response_list
            step_output_list = [judged_output.step_output for judged_output in judged_output_list]
            recorder.set_step_examples(self.generation_prompt.get_step_name(), step_output_list)
        print("ranking")
        if self.judgement_prompt is not None:
            rank_step = RankOutputStep(self.generation_prompt, judged_output_list,
                                       self.judgement_prompt, self.judgement_model_info_list).create_step(recorder)
            rank_step.record_step()
            result_output_list = rank_step.get_step_output().ordered_response_list
        else:
            result_output_list = step_output_list
        result = OrderedStepOutputList(result_output_list)
        return result


class GenerateAggregateRankCycleStep(LLMonPypeline):
    def __init__(self, generation_prompt, generation_model_info_list, aggregation_model_info_list,
                 repeat_aggregation_layer: int = 2,
                 judgement_prompt = None, judgement_model_info_list = None, max_cycles = 3, number_of_examples = 8):
        self.generation_prompt = generation_prompt
        self.generation_model_info_list = generation_model_info_list
        self.aggregation_model_info_list = aggregation_model_info_list
        self.judgement_prompt = judgement_prompt
        self.judgement_model_info_list = judgement_model_info_list
        self.repeat_aggregation_layer = repeat_aggregation_layer
        self.max_cycles = max_cycles
        self.number_of_examples = number_of_examples
        self.example_list: [JudgedOutput] = []

    def get_step_type(self) -> str:
        return STEP_TYPE_GAR

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        generation_model_info_list = [model_info.to_dict() for model_info in self.generation_model_info_list]
        aggregation_model_info_list = [model_info.to_dict() for model_info in self.aggregation_model_info_list]
        judgement_model_info_list = [model_info.to_dict() for model_info in self.judgement_model_info_list]
        result = {"generation_prompt": self.generation_prompt.get_prompt_text(),
                  "generation_model_info_list": generation_model_info_list,
                  "aggregation_model_info_list": aggregation_model_info_list,
                  "judgement_prompt": self.judgement_prompt.get_prompt_text(),
                  "judgement_model_info_list": judgement_model_info_list}
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        gar = GenerateAggregateRankStep(self.generation_prompt, self.generation_model_info_list,
                                               self.aggregation_model_info_list, self.repeat_aggregation_layer,
                                       self.judgement_prompt, self.judgement_model_info_list).create_step(recorder)
        gar.record_step()
        first_round_result_list = gar.get_step_output().ordered_response_list
        self.update_example_list(first_round_result_list, recorder)
        for i in range(1, self.max_cycles):
            gar = GenerateAggregateRankStep(self.generation_prompt, self.generation_model_info_list,
                                               self.aggregation_model_info_list, 1,
                                       self.judgement_prompt, self.judgement_model_info_list).create_step(recorder)
            recorder.set_step_examples(self.generation_prompt.get_step_name(), self.get_example_output_list())
            gar.record_step()
            result_list = gar.get_step_output().ordered_response_list
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


