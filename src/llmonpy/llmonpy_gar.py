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
            rank_step = RankOutputStep(self.generation_prompt.get_short_step_name(), judged_output_list,
                                       self.judgement_prompt, self.judgement_model_info_list).create_step(recorder)
            rank_step.record_step()
            result_output_list = rank_step.get_step_output().ordered_response_list
        else:
            result_output_list = step_output_list
        result = OrderedStepOutputList(result_output_list)
        return result



