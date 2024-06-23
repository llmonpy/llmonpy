from llmonpy_step import LLMonPyStep, TraceLogRecorderInterface


class LLMonPypeline(LLMonPyStep):
    def execute_step(self, recorder: TraceLogRecorderInterface):
        raise NotImplementedError()


