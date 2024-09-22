import copy

from llmon_pypeline import LLMonPypeline
from llmonpy.llmonpy_prompt import LLMonPyPrompt
from llmonpy.llmonpy_step import LLMONPY_OUTPUT_FORMAT_JSON, LLMonPyStepOutput
from llmonpy.llmonpy_tournament import TournamentJudgePrompt
from llmonpy_gar import GenerateAggregateRankStep


class RankerChecklistItem:
    def __init__(self, justification: str, test_question: str):
        self.justification = justification
        self.test_question = test_question

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    @staticmethod
    def from_dict(dict):
        return RankerChecklistItem(**dict)


class RankerChecklist:
    def __init__(self, checklist: [RankerChecklistItem]):
        self.checklist = checklist

    def to_dict(self):
        result = copy.copy(vars(self))
        if self.checklist is not None:
            result["checklist"] = [item.to_dict() for item in self.checklist]
        return result

    @staticmethod
    def from_dict(dict):
        return RankerChecklist(**dict)


class GenerateRankerChecklistPrompt(LLMonPyPrompt):
    criteria_text = """
    Ensure that the checklist covers all relevant aspects of the request, including but not limited to:
        1. Completeness: Does the response contain all required information and cover all key points requested?
        2. Accuracy: Is the information presented factually correct and verifiable?
        3. Relevance: Does the response directly address the user’s request without unnecessary or unrelated information?
        4. Consistency: Are all parts of the response aligned, with no contradictions or internal conflicts?
        5. Clarity: Is the response well-organized and easy to understand?
        6. Formatting: Is the response formatted as required (e.g., structure, units, or presentation)?
        7. Compliance: Does the response adhere to any specific instructions, guidelines, or restrictions (if applicable)?

    Tailor the checklist to the specific nature of the request, focusing on the most relevant aspects. Limit the
    checklist to from 1 to 3 test questions and ensure that the test questions are substantially different from each 
    other.  It two test questions are likely to produce the same result, then only one of them should be included.
    """
    prompt_text = """
    Given the request enclosed in <request-to-validate> tags, create a comprehensive and structured checklist to 
    validate the request.
    <request-to-validate>
        {{request_to_validate}}
    </request-to-validate>

    For each item in the checklist:
        1. Provide a clear justification for why this check is necessary.
        2. Formulate a specific test question that can be answered with either "Pass" or "Fail".

    {{criteria_text}}

    {% for checklist in example_list %}
        Here are examples of checklists that have been submitted by other contestants:
        # Contestant {{ loop.index }}
        "{{ checklist }}"
    {% endfor %}

    You should format your checklist as JSON in this format:
    {
        "checklist": [
            {
                "justification": "Justification for test_question",
                "test_question": "pass/fail test question"
            },
            {
                "justification": "Justification for next test_question",
                "test_question": "next pass/fail test question"
            },
        ]
    }
    """
    output_format = LLMONPY_OUTPUT_FORMAT_JSON

    def __init__(self, request_to_validate: str):
        super().__init__()
        self.request_to_validate = request_to_validate
        self.criteria_text = self.criteria_text

    def to_dict(self):
        result = super().to_dict()
        return result

    class LLMonPyOutput(LLMonPyPrompt.LLMonPyOutput):
        def __init__(self, checklist: [RankerChecklistItem] = None):
            super().__init__()
            self.checklist = checklist

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            if self.checklist is not None:
                result["checklist"] = [item.to_dict() for item in self.checklist]
            return result

        @classmethod
        def from_dict(cls, dictionary):
            if dictionary["checklist"] is not None:
                checklist = [RankerChecklistItem.from_dict(step_dict) for step_dict in dictionary["checklist"]]
                dictionary["checklist"] = checklist
            result = cls(**dictionary)
            return result

    class JudgePrompt(TournamentJudgePrompt):
        prompt_text = """
            # You are the judge

            Two checklists have been submitted to validate a request.  You must evaluate the checklists and determine
            which one is more comprehensive and structured while avoiding duplication.  The request is:
                <request-to-validate>
                    {{request_to_validate}}
                </request-to-validate>

            Some criteria to consider are:

            {{criteria_text}}

            ## Evaluation of checklist
            Given the criteria above, evaluate the checklists submitted by the contestants and determine which one is
            the best.
            ## Contestant 1
            The checklist from contestant 1 is:
            {{ contestant_1_checklist }}

            ## Contestant 2
            The steps from contestant 2 are:
            {{ contestant_2_checklist }}


            Please reply with JSON in the form: {"winner": 1} or {"winner": 2}.  Do not include any other text in your 
            response.
            """
        output_format = LLMONPY_OUTPUT_FORMAT_JSON

        def __init__(self, step_being_judged):
            super().__init__(step_being_judged)
            self.request_to_validate = step_being_judged.request_to_validate
            self.contestant_1_checklist = None
            self.contestant_2_checklist = None

        def set_contestants(self, contestant_1, contestant_2):
            self.contestant_1_checklist = contestant_1.checklist
            self.contestant_2_checklist = contestant_2.checklist

        def to_dict(self):
            result = super().to_dict()
            if self.contestant_1_checklist is not None:
                result["contestant_1_checklist"] = [item.to_dict() for item in self.contestant_1_checklist]
            if self.contestant_2_checklist is not None:
                result["contestant_2_checklist"] = [item.to_dict() for item in self.contestant_2_checklist]
            return result


class GenerateRankerChecklist(LLMonPypeline):
    class LLMonPyOutput(LLMonPyStepOutput):
        def __init__(self, checklist: RankerChecklist):
            super().__init__()
            self.checklist = checklist

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            if self.checklist is not None:
                result["checklist"] = self.checklist.to_dict()
            return result

    def __init__(self, request_to_validate: str, generate_info_list=None, aggregate_info_list=None,
                 judge_client_info_list=None):
        self.request_to_validate = request_to_validate
        self.generate_info_list = generate_info_list
        self.aggregate_info_list = aggregate_info_list
        self.judge_client_info_list = judge_client_info_list

    def get_input_dict(self, recorder):
        return {"request_to_validate": self.request_to_validate}

    def execute_step(self, recorder):
        generator_prompt = GenerateRankerChecklistPrompt(self.request_to_validate)
        judgement_prompt = GenerateRankerChecklistPrompt.JudgePrompt(generator_prompt)
        gar = GenerateAggregateRankStep(generator_prompt, self.generate_info_list, self.aggregate_info_list, 2,
                                        judgement_prompt, self.judge_client_info_list).create_step(recorder)
        gar.record_step()
        ordered_response_list = gar.get_step_output().ordered_response_list
        validation_checklist = RankerChecklist(ordered_response_list[0].step_output.checklist)
        result = GenerateRankerChecklist.LLMonPyOutput(validation_checklist)
        return result
