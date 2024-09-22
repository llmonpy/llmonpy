import copy
import json
import os
import traceback

import wandb
import weave
from weave import Dataset
from experiments.judgement_day.llmonpy_validator import GenerateValidationChecklist
from experiments.judgement_day.test_question import TestQuestion
from llm_client import MISTRAL_LARGE, GEMINI_FLASH, FIREWORKS_LLAMA3_1_405B, ANTHROPIC_SONNET, GPT4o, ANTHROPIC_HAIKU, \
    GPT4omini, FIREWORKS_LLAMA3_1_8B, GPT3_5, MISTRAL_NEMO_12B, FIREWORKS_LLAMA3_1_70B, MISTRAL_SMALL
from llmonpy.llmon_pypeline import LLMonPypeline
from llmonpy.llmonpy_step import LLMonPyStepOutput, TraceLogRecorderInterface, make_model_list, ModelTemp
from llmonpy.system_startup import llmonpy_stop, llmonpy_start
from llmonpy.trace_log import trace_log_service


class RunEvalTest(LLMonPypeline):
    class LLMonPyOutput(LLMonPyStepOutput):
        def __init__(self, test_question:TestQuestion, validator, ranker, worse_answer, invalid_answer,
                     good_answer_validated, invalid_answer_not_validated, good_answer_won_ranker):
            self.test_question = test_question
            self.validator = validator
            self.ranker = ranker
            self.worse_answer = worse_answer
            self.invalid_answer = invalid_answer
            self.good_answer_validated = good_answer_validated
            self.invalid_answer_not_validated = invalid_answer_not_validated
            self.good_answer_won_ranker = good_answer_won_ranker

        def to_dict(self):
            result = {"test_question": self.test_question.to_dict(),
                      "validator": self.validator.to_dict()}
            """result = {"test_question": self.test_question.to_dict(),
                      "validator": self.validator.to_dict(),
                      "ranker": self.ranker.to_dict(),
                      "worse_answer": self.worse_answer,
                      "invalid_answer": self.invalid_answer,
                      "good_answer_validated": self.good_answer_validated,
                      "invalid_answer_not_validated": self.invalid_answer_not_validated,
                      "good_answer_won_ranker": self.good_answer_won_ranker}
          """
            return result

    def __init__(self, test_question:TestQuestion):
        self.test_question = test_question

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        request = self.test_question.to_dict()
        result = {"test_question": request}
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        GENERATION_MODEL_INFO_LIST = make_model_list(
            ModelTemp([GPT4o, ANTHROPIC_SONNET, FIREWORKS_LLAMA3_1_70B, GEMINI_FLASH], [0.0, 0.5]))
        AGGREGATION_MODEL_INFO_LIST = make_model_list(
            ModelTemp([ANTHROPIC_HAIKU, GPT4omini, GEMINI_FLASH, FIREWORKS_LLAMA3_1_8B], [0.0, 0.5]))
        JUDGEMENT_MODEL_INFO_LIST = make_model_list(
            ModelTemp([ANTHROPIC_HAIKU, GPT4omini, GEMINI_FLASH, FIREWORKS_LLAMA3_1_8B, MISTRAL_NEMO_12B], [0.0]))
        SHORT_JUDGEMENT_MODEL_INFO_LIST = make_model_list(ModelTemp([ANTHROPIC_HAIKU, GPT4omini, GEMINI_FLASH], [0.0]))

        step_list = []
        request_str = self.test_question.generate_request()
        validation_step = GenerateValidationChecklist(request_str, GENERATION_MODEL_INFO_LIST,
                                                      AGGREGATION_MODEL_INFO_LIST,JUDGEMENT_MODEL_INFO_LIST).create_step(recorder)
        validation_step.record_step()
        validator = validation_step.get_step_output()
        result = self.LLMonPyOutput(self.test_question, validator, None, None, None,
                                    False,  False,  False)
        # create validator
        # create ranker
        # create worse answer
        # run validator on good answer
        # run ranker on worse answer
        return result


class RunEvalTestList(LLMonPypeline):
    class LLMonPyOutput(LLMonPyStepOutput):
        def __init__(self, response_list):
            self.response_list = response_list

        def to_dict(self):
            result = {"response_list": [response.to_dict() for response in self.response_list]}
            return result

    def __init__(self, eval_test_list: [RunEvalTest]):
        self.eval_test_list = eval_test_list

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        eval_test_list = [test.to_dict() for test in self.eval_test_list]
        result = {"eval_test_list": eval_test_list}
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        step_list = []
        for eval_test in self.eval_test_list:
            step = RunEvalTest(eval_test).create_step(recorder)
            step_list.append(step)
        self.run_parallel_steps(step_list)
        response_list = [step.get_step_output() for step in step_list]
        result = self.LLMonPyOutput(response_list)
        return result


def save_qbawa():
    api_key =os.environ.get("WANDB_API_KEY")
    wandb.login(key=api_key)
    weave.init("public_judgement_day")
    tourney_result_list = trace_log_service().get_tourney_results_for_step_name("GenerateValidationChecklistPrompt")
    qbawa_list = []
    for tourney in tourney_result_list:
        tourney_qbawa = tourney.generate_qbawa()
        qbawa_list.extend(tourney_qbawa)
    qbawa_list = [qbawa.to_dict() for qbawa in qbawa_list]
    dataset = Dataset(name="GenerateValidationChecklistPrompt", rows=qbawa_list)
    weave.publish(dataset)


def write_training_data():
    tourney_result_list = trace_log_service().get_tourney_results_for_step_name("GenerateValidationChecklistPrompt")
    qbawa_list = []
    for tourney in tourney_result_list:
        tourney_qbawa = tourney.generate_qbawa()
        qbawa_list.extend(tourney_qbawa)
    qbawa_list = [qbawa.to_dict() for qbawa in qbawa_list]
    file_path = "training_data.json"
    with open(file_path, "w") as file:
        json.dump(qbawa_list, file, indent=4)


if __name__ == "__main__":
    llmonpy_start()
    try:
        api_file_path = os.path.abspath(__file__)
        api_dir = os.path.dirname(api_file_path)
        file_path = api_dir + "/eval_test_data.json"
        with open(file_path, "r") as file:
            full_test_data = json.load(file)
        partial_test_data = full_test_data[:1]
        test_question_list = [TestQuestion.from_dict(test_question) for test_question in partial_test_data]
        step = RunEvalTestList(test_question_list).create_step(None)
        response = step.record_step()
        #save_qbawa()
        print("done")
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(stack_trace)
        print("Error:", str(e))
    finally:
        llmonpy_stop()
        exit(0)
