import copy
import json
import traceback
import uuid
import sys

from llmonpy.llm_client import GPT4omini, GEMINI_FLASH, FIREWORKS_MYTHOMAXL2_13B, FIREWORKS_LLAMA3_1_8B, \
    ANTHROPIC_HAIKU, MISTRAL_7B, GPT4o, ANTHROPIC_SONNET, GEMINI_PRO, FIREWORKS_LLAMA3_1_405B, MISTRAL_NEMO_12B, GPT3_5, \
    MISTRAL_LARGE, AI21_JAMBA_1_5_MINI, GEMINI_FLASH_8B, OLD_GEMINI_FLASH
from llmonpy.llmon_pypeline import LLMonPypeline
from llmonpy.llmonpy_prompt import LLMonPyPrompt, create_prompt_steps
from llmonpy.llmonpy_step import LLMonPyStepOutput, LLMONPY_OUTPUT_FORMAT_TEXT, TraceLogRecorderInterface, \
    STEP_TYPE_PYPELINE, make_model_list, ModelTemp, LlmModelInfo, TrackedOutput
from llmonpy.system_startup import llmonpy_start, llmonpy_stop


class ColaTestData:
    def __init__(self, instance_dict):
        self.id = instance_dict["id"]
        self.sentence = instance_dict["instance"]
        self.answer = instance_dict["annotations"]["grammaticality"]["majority_human"]

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    @staticmethod
    def read_from_file(file_path):
        cola_test_list = []
        with open(file_path, "r") as file:
            cola_file_dict = json.load(file)
            instance_list = cola_file_dict["instances"]
            for instance_dict in instance_list:
                cola_test = ColaTestData(instance_dict)
                cola_test_list.append(cola_test)
        return cola_test_list


class ModelPerformance:
    def __init__(self, model_description:str, right_answer_count=0, wrong_answer_count=0):
        self.model_description = model_description
        self.right_answer_count = right_answer_count
        self.wrong_answer_count = wrong_answer_count
        self.total_cost = 0

    def add_test_result(self, right, cost):
        if right:
            self.right_answer_count += 1
        else:
            self.wrong_answer_count += 1
        self.total_cost += cost


class PassFailOutput(LLMonPyStepOutput):
    def __init__(self):
        self.passed = None

    def did_pass(self):
        return self.passed

    def calc_passed(self):
        raise NotImplementedError()

    def to_dict(self):
        result = copy.copy(vars(self))
        return result


class MatchPassFailOutput(PassFailOutput):
    def __init__(self, generated_answer, correct_answer):
        super().__init__()
        self.generated_answer = generated_answer if generated_answer is None else generated_answer.strip()
        self.correct_answer = correct_answer
        self.passed = self.calc_passed()

    def calc_passed(self):
        result = False
        if self.generated_answer is not None:
            result = self.correct_answer.lower() in self.generated_answer.lower()
        return result


class ColaAnalyzeSentencePrompt(LLMonPyPrompt):
    prompt_text = """
        Can you concisely describe any grammatical errors you see in this sentence?  If you do not see any errors, 
        reply with "There are no grammatical errors in this sentence". 
        <sentence>
        {{ sentence }}
        </sentence>
        
        {% if expert_analysis_list %}
            Other experts have analyzed this sentence and they said:
            
            {% for expert in expert_analysis_list %}
                # Expert {{ loop.index }}
                "{{ expert.analysis }}"
            {% endfor %}
        {% endif %}
        What is your analysis of the sentences grammar? Do not try to correct the sentence.  Only respond with your
        analysis of the sentence grammar as it is.
    """
    system_prompt = "You are an expert at english grammar."
    output_format = LLMONPY_OUTPUT_FORMAT_TEXT

    class LLMonPyOutput(LLMonPyStepOutput):
        def __init__(self, analysis):
            super().__init__()
            self.analysis = analysis

        def to_dict(self):
            result = super().to_dict()
            return result

    def __init__(self, sentence, expert_analysis_list: [LLMonPyOutput] = None):
        super().__init__()
        self.sentence = sentence
        self.expert_analysis_list = expert_analysis_list

    def to_dict(self):
        result = super().to_dict()
        if self.expert_analysis_list is not None:
            result["expert_analysis_list"] = [analysis.to_dict() for analysis in self.expert_analysis_list]
        return result

    def output_from_string(self, string):
        result = self.LLMonPyOutput(string)
        return result


class AggregateColaPrompt(LLMonPyPrompt):
    prompt_text = """
    Several grammar experts have analyzed this sentence for grammatical errors:
    <sentence>
    {{ sentence }}
    </sentence>
    
    {% for expert in expert_analysis_list %}
        # Expert {{ loop.index }}
        "{{ expert.analysis }}"
    {% endfor %}
    
    Is the sentence grammatically correct?  Write 'Yes' if it is grammatical, 
        and 'No' if it is not.  Do not include any other text in your response.
    """
    system_prompt = "You are an expert at english grammar."
    output_format = LLMONPY_OUTPUT_FORMAT_TEXT

    def __init__(self, id, sentence, correct_answer, expert_analysis_list: [ColaAnalyzeSentencePrompt.LLMonPyOutput]):
        super().__init__()
        self.id = id
        self.sentence = sentence
        self.correct_answer = correct_answer
        self.expert_analysis_list = expert_analysis_list

    def to_dict(self):
        result = super().to_dict()
        result["expert_analysis_list"] = [analysis.to_dict() for analysis in self.expert_analysis_list]
        return result

    def output_from_string(self, string):
        result = self.LLMonPyOutput(self.id, string, self.correct_answer)
        return result

    class LLMonPyOutput(MatchPassFailOutput):
        def __init__(self, id, generated_answer, correct_answer):
            super().__init__(generated_answer, correct_answer)
            self.id = id


class ColaPrompt(LLMonPyPrompt):
    prompt_text = """
    Given the following sentence, determine if it is grammatically correct or not. Write 'Yes' if it is grammatical, 
    and 'No' if it is not.  Do not include any other text in your response. Here is the sentence:
    {{ sentence }}
    
    """
    system_prompt = "You are an expert at english grammar."
    output_format = LLMONPY_OUTPUT_FORMAT_TEXT

    def __init__(self, id, sentence, correct_answer):
        super().__init__()
        self.id = id
        self.sentence = sentence
        self.correct_answer = correct_answer

    def output_from_string(self, string):
        result = self.LLMonPyOutput(self.id, string, self.correct_answer)
        return result

    class LLMonPyOutput(MatchPassFailOutput):
        def __init__(self, id, generated_answer, correct_answer):
            super().__init__(generated_answer, correct_answer)
            self.id = id


class ColaFewShotPrompt(LLMonPyPrompt):
    prompt_text = """
    I would like you to evaluate a sentence for grammatical correctness.  Please respond with 'Yes' if the sentence is 
    grammatical, and 'No" if it is not grammatical. Here are some examples of good responses:
    
    # Start of Examples
    
    sentence:
    "Mary listens to the Grateful Dead, she gets depressed."
    response:
    Yes
    
    sentence:
    "He can not have been working."
    response:
    Yes
    
    sentence:
    "The box contained the ball from the tree."
    response:
    No
    
    sentence:
    "What the water did to the bottle was fill it."
    response:
    No
    
    sentence:
    "What the water did to the whole bottle was fill it."
    response:
    No

    sentence:
    "The tank leaked the fluid free."
    response:
    Yes

    sentence:
    "John lay the ball in the box."
    response:
    Yes
    
    sentence:
    "Most people probably consider, even though the courts didn't actually find, Klaus guilty of murder."
    response:
    Yes
    
    sentence:
    "Mary intended John to go abroad."
    response:
    No
    
    sentence:
    "Mary claimed that eating cabbage, Holly shouldn't."
    response:
    Yes
    
    # End of Examples
    
    
    Given the following sentence, determine if it is grammatically correct or not. Write 'Yes' if it is grammatical, 
    and 'No' if it is not.  Do not include any other text in your response. 
    
    sentence:
    "{{ sentence }}"
    response:

    """
    system_prompt = "You are an expert at english grammar."
    output_format = LLMONPY_OUTPUT_FORMAT_TEXT

    def __init__(self, id, sentence, correct_answer):
        super().__init__()
        self.id = id
        self.sentence = sentence
        self.correct_answer = correct_answer

    def output_from_string(self, string):
        result = self.LLMonPyOutput(self.id, string, self.correct_answer)
        return result

    class LLMonPyOutput(MatchPassFailOutput):
        def __init__(self, id, generated_answer, correct_answer):
            super().__init__(generated_answer, correct_answer)
            self.id = id


class ColaJuryStep(LLMonPypeline):
    class LLMonPyOutput(PassFailOutput):
        def __init__(self, test_data: ColaTestData, passed_list:[LlmModelInfo], failed_list:[LlmModelInfo]):
            super().__init__()
            self.test_data = test_data
            self.passed_list = passed_list
            self.failed_list = failed_list
            self.passed_count = len(passed_list)
            self.failed_count = len(failed_list)
            self.passed = self.calc_passed()

        def calc_passed(self):
            result = self.passed_count > self.failed_count
            return result

        def unanimous_but_wrong(self):
            result = self.passed_count == 0
            return result

        def to_dict(self):
            result = super().to_dict()
            result["test_data"] = self.test_data.to_dict()
            result["passed_list"] = [model_info.to_dict() for model_info in self.passed_list]
            result["failed_list"] = [model_info.to_dict() for model_info in self.failed_list]
            return result

    def __init__(self, test_data, model_info_list):
        self.test_data = test_data
        self.model_info_list = model_info_list

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        model_info_list = [model_info.to_dict() for model_info in self.model_info_list]
        result = {"test_data": self.test_data.to_dict(),
                  "model_list": model_info_list}
        return result

    def execute_cola_steps(self, recorder: TraceLogRecorderInterface) -> [TrackedOutput]:
        raise NotImplementedError

    def execute_step(self, recorder: TraceLogRecorderInterface):
        response_list = self.execute_cola_steps(recorder)
        passed_list = []
        failed_list = []
        for response in response_list:
            if response.step_output.did_pass():
                passed_list.append(response)
            else:
                failed_list.append(response)
        result = self.LLMonPyOutput(self.test_data, passed_list, failed_list)
        return result


class FewShotColaJuryStep(ColaJuryStep):
    def __init__(self, test_data, model_info_list):
        super().__init__(test_data, model_info_list)

    def execute_cola_steps(self, recorder: TraceLogRecorderInterface) -> [TrackedOutput]:
        judge_prompt = ColaFewShotPrompt(self.test_data.id, self.test_data.sentence, self.test_data.answer)
        step_list = create_prompt_steps(recorder, judge_prompt, self.model_info_list)
        self.run_parallel_steps(step_list)
        response_list = []
        for step in step_list:
            step_output = step.get_step_output()
            tracked_output = TrackedOutput(recorder.get_step_id(), step_output, step.get_model_info())
            response_list.append(tracked_output)
        return response_list


class AnalyzeAggregateColaJuryStep(ColaJuryStep):
    def __init__(self, test_data, model_info_list):
        super().__init__(test_data, model_info_list)

    def execute_analyze_steps(self, recorder: TraceLogRecorderInterface,
                              analysis_list:[ColaAnalyzeSentencePrompt.LLMonPyOutput] = None) -> [ColaAnalyzeSentencePrompt.LLMonPyOutput]:
        analyze_prompt = ColaAnalyzeSentencePrompt(self.test_data.sentence, analysis_list)
        analyze_step_list = create_prompt_steps(recorder, analyze_prompt, self.model_info_list)
        self.run_parallel_steps(analyze_step_list)
        analysis_list:[ColaAnalyzeSentencePrompt.LLMonPyOutput] = []
        for step in analyze_step_list:
            step_output = step.get_step_output()
            analysis_list.append(step_output)
        return analysis_list

    def execute_cola_steps(self, recorder: TraceLogRecorderInterface) -> [TrackedOutput]:
        analysis_list:[ColaAnalyzeSentencePrompt.LLMonPyOutput] = self.execute_analyze_steps(recorder)
        analysis_list:[ColaAnalyzeSentencePrompt.LLMonPyOutput] = self.execute_analyze_steps(recorder, analysis_list)
        judge_prompt = AggregateColaPrompt(self.test_data.id, self.test_data.sentence, self.test_data.answer, analysis_list)
        step_list = create_prompt_steps(recorder, judge_prompt, self.model_info_list)
        self.run_parallel_steps(step_list)
        response_list = []
        for step in step_list:
            step_output = step.get_step_output()
            tracked_output = TrackedOutput(recorder.get_step_id(), step_output, step.get_model_info())
            response_list.append(tracked_output)
        return response_list


class ColaPypeLine(LLMonPypeline):
    class LLMonPyOutput(LLMonPyStepOutput):
        def __init__(self, response_list: [ColaJuryStep.LLMonPyOutput]):
            self.response_list = response_list

        def to_dict(self):
            result = {"response_list": [response.to_dict() for response in self.response_list]}
            return result

    def __init__(self, cola_test_list, model_info_list):
        self.cola_test_list = cola_test_list
        self.model_info_list = model_info_list

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        model_info_list = [model_info.to_dict() for model_info in self.model_info_list]
        result = {"cola_test_list": [cola_test.to_dict() for cola_test in self.cola_test_list],
                  "model_list": model_info_list}
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        step_list = []
        for cola_test in self.cola_test_list:
            step = AnalyzeAggregateColaJuryStep( cola_test, self.model_info_list).create_step(recorder)
            step_list.append(step)
        self.run_parallel_steps(step_list)
        response_list = [step.get_step_output() for step in step_list]
        result = self.LLMonPyOutput(response_list)
        return result

FIVE_SMALL_MODEL_JUDGE_LIST =[ANTHROPIC_HAIKU, GPT4omini, GPT4o, GEMINI_FLASH, FIREWORKS_LLAMA3_1_8B]
ALL_SMALL_MODEL_JUDGE_LIST = [ANTHROPIC_HAIKU, GPT4omini, GEMINI_FLASH, FIREWORKS_LLAMA3_1_8B, FIREWORKS_MYTHOMAXL2_13B, MISTRAL_7B, GPT3_5]
LARGE_MODEL_JUDGE_LIST = [GPT4o, ANTHROPIC_SONNET, FIREWORKS_LLAMA3_1_405B, GEMINI_FLASH, MISTRAL_LARGE]
MIXED_MODEL_LIST_1 = [ANTHROPIC_HAIKU, GPT4omini, GEMINI_FLASH, GPT4o, ANTHROPIC_SONNET]
MIXED_MODEL_LIST_2 = [ANTHROPIC_HAIKU, GPT4omini, GEMINI_FLASH, FIREWORKS_LLAMA3_1_8B, FIREWORKS_MYTHOMAXL2_13B, MISTRAL_7B, GPT3_5, GEMINI_PRO, GPT4o, ANTHROPIC_SONNET, MISTRAL_LARGE]
ALT_FIVE_SMALL_MODEL_JUDGE_LIST =[GEMINI_FLASH]


if __name__ == "__main__":
    llmonpy_start()
    try:
        cola_file_path="data/cola.json"
        if len(sys.argv) >= 2:
            cola_file_path = sys.argv[1]
        cola_test_list = ColaTestData.read_from_file(cola_file_path)
        model_info_list = make_model_list(ModelTemp(FIVE_SMALL_MODEL_JUDGE_LIST, [0.0]))
        subset = cola_test_list[:100]
        cola_pipeline_step = ColaPypeLine(subset, model_info_list).create_step(None)
        cola_pipeline_step.record_step()
        response_list = cola_pipeline_step.get_step_output().response_list
        passed_count = 0
        failed_count = 0
        failed_list = []
        for cola_test in response_list:
            if cola_test.did_pass():
                print("Passed")
                passed_count += 1
            else:
                print("Failed test for sentence: " + cola_test.test_data.sentence)
                failed_list.append(cola_test)
                failed_count += 1
        print("Passed: " + str(passed_count))
        print("Failed: " + str(failed_count))
        failed_list = [response.to_dict() for response in failed_list]
        results_path = cola_file_path.replace(".json", "_results.json")
        with open(results_path, "w") as results_file:
            json.dump(failed_list, results_file, indent=4)
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(stack_trace)
        print("Error:", str(e))
    finally:
        llmonpy_stop()
        exit(0)
