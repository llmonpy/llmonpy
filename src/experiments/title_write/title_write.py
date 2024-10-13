import copy

from llmonpy.llmonpy_prompt import LLMonPyPrompt
from llmonpy.llmonpy_step import LLMONPY_OUTPUT_FORMAT_TEXT
from llmonpy_tournament import TournamentJudgePrompt


class GenerateNameSuggestionPrompt(LLMonPyPrompt):
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
    system_prompt = """
    You are a highly creative naming expert with a talent for crafting unique, evocative, and memorable names. 
    Your task is to generate names that are:

    1. Descriptive: They should convey the essence or key features of the thing being named.
    2. Creative: Use wordplay, alliteration, rhymes, or unexpected combinations to make the names stand out.
    3. Memorable: The names should be easy to remember and have a strong impact.
    """
    output_format = LLMONPY_OUTPUT_FORMAT_TEXT

    def __init__(self, type_of_thing_to_name: str, description_of_thing_to_name: str):
        super().__init__()
        self.type_of_thing_to_name = type_of_thing_to_name
        self.description_of_thing_to_name = description_of_thing_to_name

    def to_dict(self):
        result = super().to_dict()
        return result

    class LLMonPyOutput(LLMonPyPrompt.LLMonPyOutput):
        def __init__(self, suggested_name: str):
            super().__init__()
            self.suggested_name = suggested_name

        def to_dict(self):
            result = copy.deepcopy(vars(self))
            self.suggested_name = self.suggested_name
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
