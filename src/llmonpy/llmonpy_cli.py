import argparse

from llmonpy.llm_client import get_active_llm_clients, MISTRAL_7B
from llmonpy.system_startup import llmonpy_start, llmonpy_stop
from llmonpy.llmonpy_execute import run_step
from llmonpy.prompt import LLMonPyPromptEvaluator
from llmonpy.example.test_cycle import GenerateNameCycle
from llmonpy.example.test_prompts import TestLLMonPyPrompt
from llmonpy.example.test_tourney import GenerateNamePypeline


def llmonpy_cli():
    parser = argparse.ArgumentParser(description='Run specific functions from the command line.')
    parser.add_argument('function', choices=['models', 'prompt', 'tourney', 'cycle'],
                        help='The function to run.')
    args = parser.parse_args()
    llmonpy_start()
    try:
        if args.function == 'models':
            model_list = get_active_llm_clients()
            if len(model_list) == 0:
                print("No models are active.")
            else:
                for model in model_list:
                    print(model.model_name)
        elif args.function == 'prompt':
            step = LLMonPyPromptEvaluator(MISTRAL_7B, TestLLMonPyPrompt("LLMonPy"))
            result, recorder = run_step(step)
            print(result.to_json())
        elif args.function == 'tourney':
            step = GenerateNamePypeline()
            result, recorder = run_step(step)
            print(result.to_json())
        elif args.function == 'cycle':
            step = GenerateNameCycle()
            result, recorder = run_step(step)
            print(result.to_json())
    finally:
        llmonpy_stop()
        exit(0)

if __name__ == "__main__":
    llmonpy_cli()