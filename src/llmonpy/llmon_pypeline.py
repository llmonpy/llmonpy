import copy

from llmonpy.llmonpy_step import LLMonPyStep, TraceLogRecorderInterface, STEP_TYPE_PYPELINE, \
    get_step_name_from_class_hierarchy


class LLMonPypeline:
    def execute_step(self, recorder: TraceLogRecorderInterface):
        raise NotImplementedError()

    def get_step_type(self) -> str:
        return STEP_TYPE_PYPELINE

    def get_step_name(self):
        result = get_step_name_from_class_hierarchy(self.__class__)
        return result

    def get_short_step_name(self):
        result = self.__class__.__name__
        return result

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        raise NotImplementedError()

    def to_dict(self):
        result = copy.copy(vars(self))
        return result

    def create_step(self):
        return LLMonPypelineRunner(self)


class LLMonPypelineRunner(LLMonPyStep):
    def __init__(self, pypeline: LLMonPypeline):
        self.pypeline = pypeline

    def get_step_name(self):
        return self.pypeline.get_step_name()

    def get_step_type(self) -> str:
        return self.pypeline.get_step_type()

    def get_input_dict(self, recorder: TraceLogRecorderInterface):
        super_result = super().get_input_dict(recorder)
        result = self.pypeline.get_input_dict(recorder)
        result.update(super_result)
        return result

    def execute_step(self, recorder: TraceLogRecorderInterface):
        result = None
        try:
            result, _ = self.pypeline.execute_step(recorder)
        except Exception as e:
            recorder.log_exception(e)
            raise e
        return result, recorder

