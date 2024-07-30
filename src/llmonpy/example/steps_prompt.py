import argparse
import copy
import json
import uuid

from llmonpy.jsony import jsony_to_json
from llmonpy.llm_client import MISTRAL_7B, filter_clients_that_didnt_start, GPT4o, MISTRAL_LARGE, GEMINI_PRO, GEMINI_FLASH, \
    ANTHROPIC_SONNET, ANTHROPIC_HAIKU, MISTRAL_8X22B, GPT4omini
from llmonpy.llmon_pypeline import LLMonPypeline
from llmonpy.llmonpy_execute import do_llmonpy_step, run_step
from llmonpy.llmonpy_step import TraceLogRecorderInterface, LLMONPY_OUTPUT_FORMAT_JSON, LlmModelInfo, make_model_list, \
    ModelTemp
from llmonpy.llmonpy_tournament import TournamentJudgePrompt, LLMonPyTournament, AdaptiveICLCycle
from llmonpy.llmonpy_prompt import LLMonPyPrompt, LLMonPyPromptExecutor
from llmonpy.system_startup import llmonpy_start, llmonpy_stop


class ProjectStep:
    def __init__(self, description_of_code):
        self.description_of_code = description_of_code

    def to_dict(self):
        return copy.deepcopy(vars(self))

    @staticmethod
    def from_dict(dictionary):
        return ProjectStep(**dictionary)


class GenerateProjectSteps(LLMonPyPrompt):
    prompt_text = """
    I would like you to break down this project into a list of steps.  Here is the project description:

    {{ project_description }}
    
    The list of steps that you generate should assume that you start with this code:
    
    {{ starting_point }}
    
    After you have written the code for each step, the code will be tested with this input:
    
    {{ test_case }}
    
    Please make a step-by-step plan to take this project description to code I can run on my computer.  Each step should
    describe code you will write in that step that will get you closer to the final code.   Do not include any steps 
    that do not involve writing code.  Assume that the environment that will run the code is setup and is working.  
    The steps should be as small as you can make them, but "complete" in the sense that they do compile and run.  Don't 
    use regular expressions in this project.
    
    Present these steps in JSON format. For each step, provide a description of the code that you will write for that 
    step. Here is an example of the result of this output:
    
    { "step_list": [{
      "description_of_code": "Open and read the input file",
    },
    {
      "description_of_code": "Parse the input file and break it into separate templates"
    }]}
    
    {% if example_list  %}
        This is a list of examples of good responses to this request.  The first example is the worst response and the
        last one is the best response.

            {% for example in example_list %}
                Good response to this request:
                { "step_list": {{ example.step_list | tojson }} }
            {% endfor %}
    {% endif %}

    """
    system_prompt = """
    You are an expert software developer.  As an expert, you know how to break a project down into short, manageable 
    steps and then translate those steps into code
    """
    output_format = LLMONPY_OUTPUT_FORMAT_JSON

    def __init__(self, project_description, starting_point, test_case):
        super().__init__()
        self.project_description = project_description
        self.starting_point = starting_point
        self.test_case = test_case

    class LLMonPyOutput(LLMonPyPrompt.LLMonPyOutput):
        def __init__(self, step_list=None):
            super().__init__()
            self.step_list = step_list

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            if self.step_list is not None:
                result["step_list"] = [step.to_dict() for step in self.step_list]
            return result

        @classmethod
        def from_dict(cls, dictionary):
            if dictionary["step_list"] is not None:
                step_list = [ProjectStep.from_dict(step_dict) for step_dict in dictionary["step_list"]]
                dictionary["step_list"] = step_list
            result = cls(**dictionary)
            return result

    class JudgePrompt(TournamentJudgePrompt):
        prompt_text = """
            # You are the judge

            Two people have been asked to break down a software development project into smaller tasks. The project description is:
            
            ## Project Description
            {{ project_description }}
            
            
            ## Evaluation of step list
            You should consider the following when evaluating each step:
            1. The project only covers writing the code.  No step should involve product or project management, 
            marketing, or any other non-coding task.
            2. Each step must build on prior steps and not require steps that come later in the list.
            3. Each step must describe code that will be written in that step that will get you closer to the final code. 
            4. The steps must be as small as you can make them, but "complete" in the sense that they do compile and run.
            5. The steps must implement all the features from the project description.
            
            ## Contestant 1
            The steps from contestant 1 are:
            {{ contestant_1_step_list }}
            
            ## Contestant 2
            The steps from contestant 2 are:
            {{ contestant_2_step_list }}


            Please reply with JSON in the form: {"winner": 1} or {"winner": 2}.  Do not include any other text in your 
            response.
            """
        system_prompt = """
            You are an expert software developer.  As an expert, you know how to break a project down into short, manageable 
            steps and then translate those steps into code            
            """
        output_format = LLMONPY_OUTPUT_FORMAT_JSON

        def __init__(self, step_being_judged):
            super().__init__(step_being_judged)
            self.project_description = step_being_judged.project_description
            self.contestant_1_step_list = None
            self.contestant_2_step_list = None

        def set_contestants(self, contestant_1, contestant_2):
            self.contestant_1_step_list = contestant_1.step_list
            self.contestant_2_step_list = contestant_2.step_list

        def to_dict(self):
            result = super().to_dict()
            if self.contestant_1_step_list is not None:
                result["contestant_1_step_list"] = [step.to_dict() for step in self.contestant_1_step_list]
            if self.contestant_2_step_list is not None:
                result["contestant_2_step_list"] = [step.to_dict() for step in self.contestant_2_step_list]
            return result

        @classmethod
        def from_dict(cls, dictionary):
            if dictionary["contestant_1_step_list"] is not None:
                contestant_1_step_list = [ProjectStep.from_dict(step_dict) for step_dict in dictionary["contestant_1_step_list"]]
                dictionary["contestant_1_step_list"] = contestant_1_step_list
            if dictionary["contestant_2_step_list"] is not None:
                contestant_2_step_list = [ProjectStep.from_dict(step_dict) for step_dict in dictionary["contestant_2_step_list"]]
                dictionary["contestant_2_step_list"] = contestant_2_step_list
            result = cls(**dictionary)
            return result


class GenerateProjectStepsTourney(LLMonPypeline):
    def __init__(self, project_description, starting_point, test_case):
        super().__init__()
        self.project_description = project_description
        self.starting_point = starting_point
        self.test_case = test_case

    def execute_step(self, recorder: TraceLogRecorderInterface):
        client_list = [GPT4o, GPT4omini, GEMINI_PRO, GEMINI_FLASH, ANTHROPIC_SONNET, MISTRAL_7B, ANTHROPIC_HAIKU]
        client_info_list = make_model_list(ModelTemp(client_list, [0.0, 0.5]))
        judge_client_info_list = make_model_list(ModelTemp([GPT4omini, GEMINI_FLASH, MISTRAL_7B, MISTRAL_8X22B,
                                                             ANTHROPIC_HAIKU],0.0))
        generator_prompt = GenerateProjectSteps(self.project_description, self.starting_point, self.test_case)
        judgement_prompt = GenerateProjectSteps.JudgePrompt(generator_prompt)
        tournament = LLMonPyTournament(generator_prompt, client_info_list, judgement_prompt, judge_client_info_list)
        result_list, _ = do_llmonpy_step(tournament, recorder)
        return result_list[0].step_output, recorder


class GenerateProjectStepsCycle(LLMonPypeline):
    def __init__(self, project_description, starting_point, test_case):
        super().__init__()
        self.project_description = project_description
        self.starting_point = starting_point
        self.test_case = test_case

    def execute_step(self, recorder: TraceLogRecorderInterface):
        client_list = [GPT4o, GPT4omini, GEMINI_FLASH, ANTHROPIC_SONNET, MISTRAL_7B, ANTHROPIC_HAIKU]
        client_info_list = make_model_list(ModelTemp(client_list, 0.0), ModelTemp(client_list,0.5))
        judge_client_info_list = make_model_list(ModelTemp([GPT4omini, GEMINI_FLASH, MISTRAL_7B, MISTRAL_8X22B,
                                                             ANTHROPIC_HAIKU],0.0))
        generator_prompt = GenerateProjectSteps(self.project_description, self.starting_point, self.test_case)
        judgement_prompt = GenerateProjectSteps.JudgePrompt(generator_prompt)
        cycle = AdaptiveICLCycle(generator_prompt, client_info_list, judgement_prompt,
                                 judge_client_info_list, 5, 3)
        result_list, _ = cycle.execute_step(recorder)
        return result_list[0], recorder


def run_prompt(project_description, starting_point, test_case):
    print("run prompt")
    model_info = LlmModelInfo(MISTRAL_7B.model_name)
    step = LLMonPyPromptExecutor(GenerateProjectSteps(project_description, starting_point, test_case), model_info)
    result, recorder = run_step(step)
    print(result.to_json())


def run_tourney(project_description, starting_point, test_case):
    print("Running tourney...")
    step = GenerateProjectStepsTourney(project_description, starting_point, test_case)
    result, recorder = run_step(step)
    print(result.to_json())


def run_cycle(project_description, starting_point, test_case):
    print("Running cycle...")
    step = GenerateProjectStepsCycle(project_description, starting_point, test_case)
    result, recorder = run_step(step)
    print(result.to_json())


def main():
    parser = argparse.ArgumentParser(description='Run specific functions from the command line.')
    parser.add_argument('function', choices=['prompt', 'tourney', 'cycle'],
                        help='The function to run.')
    parser.add_argument('filename', help='The name of the file to read.')
    args = parser.parse_args()
    try:
        with open(args.filename, 'r') as file:
            file_content = file.read()
            json_input_string = jsony_to_json(file_content)
            json_input = json.loads(json_input_string)
    except FileNotFoundError:
        print(f"File '{args.filename}' not found.")
        return

    if args.function == 'prompt':
        run_prompt(project_description=json_input["project_description"], starting_point=json_input["starting_point"],
                   test_case=json_input["test_case"])
    elif args.function == 'tourney':
        run_tourney(project_description=json_input["project_description"], starting_point=json_input["starting_point"],
                   test_case=json_input["test_case"])
    elif args.function == 'cycle':
        run_cycle(project_description=json_input["project_description"], starting_point=json_input["starting_point"],
                   test_case=json_input["test_case"])


if __name__ == "__main__":
    llmonpy_start()
    try:
        main()
    except Exception as e:
        print("exception:" + str(e))
    finally:
        llmonpy_stop()
        exit(0)