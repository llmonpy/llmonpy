from llmonpy_step import LLMonPyStep


class LLMonPypeline(LLMonPyStep):
    def execute_step(self, recorder):
        raise NotImplementedError()


