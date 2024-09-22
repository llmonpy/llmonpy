import copy


class TestQuestion:
    def __init__(self, id, context, question, good_answer):
        self.id = id
        self.context = context
        self.question = question
        self.good_answer = good_answer

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    def generate_request(self):
        result = "Given this context: " +  self.context + " \n\n answer this question:" + self.question
        return result

    @classmethod
    def from_dict(cls, dictionary):
        result = cls(**dictionary)
        return result