import json

from llmonpy.llmonpy_step import LlmModelInfo


class ClassifierConfig:
    def __init__(self, team_list:[LlmModelInfo], persona: str, intro: str, policies: str, examples: str):
        self.team_list = team_list
        self.persona = persona
        self.intro = intro
        self.policies = policies
        self.examples = examples


    def to_dict(self):
        result = {"team_list": [model.to_dict() for model in self.team_list], "persona": self.persona,
                  "intro": self.intro, "policies": self.policies, "examples": self.examples}
        return result

    @staticmethod
    def from_dict(dict):
        team_list = [LlmModelInfo.from_dict(model) for model in dict["team_list"]]
        return ClassifierConfig(team_list, dict["persona"], dict["intro"], dict["policies"], dict["examples"])

    @staticmethod
    def from_file(file_path):
        with open(file_path, "r") as file:
            dict = json.load(file)
            result = ClassifierConfig.from_dict(dict)
        return result