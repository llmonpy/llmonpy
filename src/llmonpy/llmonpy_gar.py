from llmonpy.llmon_pypeline import LLMonPypeline


class GenerateAggregateRank(LLMonPypeline):
    def __init__(self, generation_prompt, generation_client_list, aggregation_client_list, rank_client_list):
        self.generation_prompt = generation_prompt
        self.generation_client_list = generation_client_list
        self.aggregation_client_list = aggregation_client_list
        self.rank_client_list = rank_client_list
