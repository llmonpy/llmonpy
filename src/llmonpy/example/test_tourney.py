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
import uuid

from llmonpy.llm_client import GPT4o, MISTRAL_LARGE, GEMINI_PRO, GEMINI_FLASH, ANTHROPIC_SONNET, MISTRAL_7B, \
    ANTHROPIC_HAIKU, \
    MISTRAL_8X22B, ANTHROPIC_OPUS, filter_clients_that_didnt_start, GPT4omini, MISTRAL_SMALL
from llmonpy.llmon_pypeline import LLMonPypeline
from llmonpy.llmonpy_execute import do_llmonpy_step, run_step
from llmonpy.llmonpy_step import TraceLogRecorderInterface, LLMONPY_OUTPUT_FORMAT_JSON, ModelTemp, make_model_list
from llmonpy.llmonpy_tournament import LLMonPyTournament, TournamentJudgePrompt
from llmonpy.llmonpy_prompt import create_prompt_steps, LLMonPyPrompt
from llmonpy.system_startup import llmonpy_start, llmonpy_stop


class NameIterativeRefinementTournamentPrompt(LLMonPyPrompt):
    prompt_text = """
            I need to name an exciting new technique for responding to requests that are given to LLMs to respond to.  
            It is well known LLMs give better responses when shown given examples of good responses.  I have created a 
            way to give LLMs an example of a good answer for the exact request they are responding too.  Given a 
            request, I ask a large group of different LLMs to respond.  Then I use another set of LLMs to judge and 
            rank the responses. The winning response is determined by majority vote of the LLM judges.  Then there is 
            another round of the tournament to generate a response, but the best results of the last round are given as 
            example of good responses. The 2nd round responses are much, much better!  The same LLMs as judge is used to
            pick a winner of the 2nd round.  Then the judges are used again to compare the winner of the first round 
            against the winner of the 2nd round.  If the 2nd round wins, the cycle is repeated but with the winner of 
            the 2nd round as the example of a good response. This continues as long as the responses keep getting better.
            This is an example of "adaptive in-context learning (ICL)".  ICL is also called "few-shot prompting". AdaptiveICL
            does not usually include multiple rounds of tournaments.
            
            Examples of good names for algorithms are: 
            
            1. "Artificial Intelligence": for what is essentially linear algebra.  
            2. "Genetic Algorithms": simulate the process of evolution to solve optimization problems
            3. "Quicksort": a sorting algorithm that is not the fastest but is very fast in practice.
            4. "Ant Colony Optimization": Inspired by the foraging behavior of ants, this algorithm is used for solving 
                combinatorial optimization problems. Ants deposit pheromones to mark paths, which influences the 
                behavior of other ants, leading to optimal solutions over time.
            5. YOLO (You Only Look Once): A real-time object detection algorithm that processes images in a single pass 
                through the network, making it faster and more efficient. The catchy name emphasizes the algorithm's 
                efficiency and speed.
            6. PageRank: Developed by Google founders Larry Page and Sergey Brin, this algorithm ranks web pages in 
                search engine results. The name is a play on Larry Page's name and the idea of ranking pages.
            7. "Deep Dream": a neural network visualization technique that generates surreal and dream-like images.
            
            {% if example_list  %}
            Improve on these examples of good names for this technique: 
                {% for example in example_list %}
                    "{{ example.name }}"
                {% endfor %}
            {% endif %}

            I would like you to suggest a name for this method of getting better responses from a LLM.  
            Please reply with JSON in the form:
            {"name": "your name suggestion"}.  Do not include any other text in your response.
            """
    system_prompt="""
        You are a genius as generating names that are descriptive, but punchy and positive.  The names you generate 
        don't sound too technical or boring.
        """
    output_format = LLMONPY_OUTPUT_FORMAT_JSON

    def __init__(self):
        super().__init__()

    def to_dict(self):
        result = copy.deepcopy(vars(self))
        return result

    class LLMonPyOutput(LLMonPyPrompt.LLMonPyOutput):
        def __init__(self, name=None):
            super().__init__()
            self.name = name

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            return result

        @classmethod
        def from_dict(cls, dictionary):
            result = cls(**dictionary)
            return result

    class JudgePrompt(TournamentJudgePrompt):
        prompt_text = """
            I need you to judge the name suggestion for a new prompting technique.  The name should be descriptive, 
            but punchy and positive.  The name should not sound too technical or boring. The name should be easy to 
            remember.  The name should be easy to spell.  The name should be easy to say. Examples of good comparisons are:
            
            Artificial Intelligence vs. PatternSolver:  Artificial Intelligence is the winner
            GenOpt vs. Genetic Algorithms:  Genetic Algorithms is the winner
            Quicksort vs. EffiSort:  Quicksort is the winner
            OnePassDetect vs. YOLO:  YOLO is the winner
            PageRank vs TopPage: PageRank is the winner
            SurrealVis vs Deep Dream: Deep Dream is the winner

            Given these instructions, which do you think is the better name:
            
            Candidate 1: {{ contestant_1_name }} vs Candidate 2: {{ contestant_2_name }}
            
            Please reply with this JSON if Candidate 1 is the winner : {"winner": 1} or with this JSON if
            Candidate 2 is the winner: {"winner": 2}.  Do not include any other text in your response.
            """
        system_prompt = """
            You are an expert at following instructions to judge names.
            """
        output_format = LLMONPY_OUTPUT_FORMAT_JSON

        def __init__(self, prompt_being_judged):
            super().__init__(prompt_being_judged)
            self.contestant_1_name = None
            self.contestant_2_name = None

        def set_contestants(self, contestant_1, contestant_2):
            self.contestant_1_name = contestant_1.name
            self.contestant_2_name = contestant_2.name

        def to_dict(self):
            result = super().to_dict()
            return result

        @classmethod
        def from_dict(cls, dictionary):
            result = cls(**dictionary)
            return result


class GenerateNamePypeline(LLMonPypeline):
    def __init__(self):
        pass

    def execute_step(self, recorder: TraceLogRecorderInterface):
        client_list = [GPT4omini, GEMINI_FLASH, ANTHROPIC_SONNET, MISTRAL_7B, ANTHROPIC_HAIKU]
        client_info_list = make_model_list(ModelTemp(client_list, [0.0, 0.5]))
        judge_client_info_list = make_model_list(ModelTemp([GPT4omini, GEMINI_FLASH, MISTRAL_7B,
                                                            MISTRAL_SMALL, ANTHROPIC_HAIKU],0.0))
        generator_prompt = NameIterativeRefinementTournamentPrompt()
        judgement_prompt = NameIterativeRefinementTournamentPrompt.JudgePrompt(generator_prompt)
        tournament = LLMonPyTournament(generator_prompt, client_info_list, judgement_prompt, judge_client_info_list)
        result_list, _ = do_llmonpy_step(tournament, recorder)
        for result in result_list:
            print("name:" + result.step_output.name + " score: " + str(result.victory_count))
        return result_list[0].step_output, recorder


if __name__ == "__main__":
    llmonpy_start()
    print("Running Test Tourney")
    try:
        step = GenerateNamePypeline()
        result, recorder = run_step(step)
        print(result.to_json())
    except Exception as e:
        print(str(e))
    finally:
        llmonpy_stop()
        exit(0)

