import argparse
import copy
import json
import uuid

from jsony import jsony_to_json
from llm_client import MISTRAL_7B
from llmonpy_tournament import TournamentJudgePrompt
from prompt import LLMonPyPrompt, LLMonPyPromptEvaluator
from system_startup import llmonpy_start, llmonpy_stop
from trace_log import trace_log_service


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
    json_output = True

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
            1. The project only covers writing the code.  No step should involve product or project management, marketing, or any other non-coding task.
            2. Each step should build on prior steps and not require steps that come later in the list.
            3. Each step should describe code that will be written in that step that will get you closer to the final code. 
            4. The steps should be as small as you can make them, but "complete" in the sense that they do compile and run.
            
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
        json_output = True

        def __init__(self, name_of_step_being_judged, project_description):
            super().__init__(name_of_step_being_judged)
            self.project_description = project_description
            self.contestant_1_step_list = None
            self.contestant_2_step_list = None

        def set_contestants(self, contestant_1, contestant_2):
            self.contestant_1_step_list = contestant_1.step_list
            self.contestant_2_step_list = contestant_2.step_list

        def to_dict(self):
            result = copy.deepcopy(vars(self))
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


def run_prompt(project_description, starting_point, test_case):
    trace_id = str(uuid.uuid4())
    print("run prompt")
    step = LLMonPyPromptEvaluator(MISTRAL_7B, GenerateProjectSteps(project_description, starting_point, test_case))
    recorder = trace_log_service().create_root_recorder(trace_id, trace_id, None, step)
    result, _ = step.execute_step(recorder)
    recorder.finish_child_step(result)
    print(result.to_json())


def run_tourney():
    print("Running tourney...")


def run_cycle():
    print("Running cycle...")


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
        run_tourney()
    elif args.function == 'cycle':
        run_cycle()


if __name__ == "__main__":
    llmonpy_start()
    try:
        main()
    except Exception as e:
        print("exception:" + str(e))
    finally:
        llmonpy_stop()
        exit(0)