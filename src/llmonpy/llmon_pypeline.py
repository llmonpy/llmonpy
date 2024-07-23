from llmonpy.llmonpy_step import LLMonPyStep, TraceLogRecorderInterface, STEP_TYPE_PYPELINE


class LLMonPypeline(LLMonPyStep):
    def execute_step(self, recorder: TraceLogRecorderInterface):
        raise NotImplementedError()

    def get_step_type(self) -> str:
        return STEP_TYPE_PYPELINE


