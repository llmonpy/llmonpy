import argparse
import json
import sys
import traceback

from ratellmiter.rate_llmiter import get_rate_limiter_monitor

from llmonpy.llm_client import get_active_llm_clients, MISTRAL_7B, GPT4omini, FIREWORKS_LLAMA3_1_8B, \
    FIREWORKS_MYTHOMAXL2_13B, FIREWORKS_GEMMA2_9B, FIREWORKS_LLAMA3_1_405B, FIREWORKS_LLAMA3_1_70B, GPT4o, GPT3_5
from llmonpy.system_startup import llmonpy_start, llmonpy_stop
from llmonpy.llmonpy_execute import run_step
from llmonpy.llmonpy_prompt import LLMonPyPromptRunner
from llmonpy.example.test_cycle import GenerateNameCycle
from llmonpy.example.test_prompts import TestLLMonPyPrompt
from llmonpy.example.test_tourney import GenerateNamePypeline
from llmonpy.llmonpy_step import LlmModelInfo, make_model_list, ModelTemp
from llmonpy.example.test_gar import GenerateNameGar
from llmonpy.trace_log import trace_log_service


def llmonpy_cli():
    parser = argparse.ArgumentParser(description='Run specific functions from the command line.')
    parser.add_argument('function', choices=['models', 'prompt', 'tourney', 'cycle', 'gar', 'qbawa_list',
                                             'qbawa', 'llmiter'],
                        help='The function to run.')
    parser.add_argument('-name', type=str, help='name argument')
    parser.add_argument('-file', type=str, help='file argument')
    parser.add_argument('-lines', type=str, help='ex: iroef, i=issued, r=requests, o=overflow, e=exceptions, f=finished')
    args = parser.parse_args()
    llmonpy_start()
    model_list = [FIREWORKS_LLAMA3_1_8B, FIREWORKS_MYTHOMAXL2_13B, GPT4o, GPT3_5, GPT4omini]
    first_round_info_list = make_model_list(ModelTemp(model_list, [0.0, 0.75]))
    aggregate_info_list = make_model_list(ModelTemp(model_list,[0.0, 0.75]))
    judge_client_info_list = make_model_list(ModelTemp(model_list, 0.0))
    try:
        if args.function == 'models':
            model_list = get_active_llm_clients()
            if len(model_list) == 0:
                print("No models are active.")
            else:
                for model in model_list:
                    print(model.model_name)
        elif args.function == 'prompt':
            print("running prompt test")
            model_list = get_active_llm_clients()
            model_info = LlmModelInfo(model_list[0].model_name)
            step = LLMonPyPromptRunner(None, TestLLMonPyPrompt("LLMonPy"), model_info)
            result = run_step(step)
            print(result.to_json())
        elif args.function == 'tourney':
            print("running tourney test")
            step = GenerateNamePypeline(client_info_list=first_round_info_list, judge_client_info_list=judge_client_info_list).create_step(None)
            result = run_step(step)
            print(result.to_json())
        elif args.function == 'cycle':
            print("running cycle test")
            step = GenerateNameCycle(first_round_info_list=first_round_info_list, aggregate_info_list=aggregate_info_list, judge_client_info_list=judge_client_info_list).create_step(None)
            result = run_step(step)
            print(result.to_json())
        elif args.function == 'gar':
            print("running gar test")
            step = GenerateNameGar(generate_info_list=first_round_info_list, aggregate_info_list=aggregate_info_list, judge_client_info_list=judge_client_info_list).create_step(None)
            result = run_step(step)
            print(result.to_json())
        elif args.function == 'qbawa_list':
            step_name_list = trace_log_service().get_tourney_step_name_list()
            print("Step names with QBaWa data:")
            for step_name in step_name_list:
                print(step_name)
        elif args.function == 'qbawa':
            step_name = args.name
            if step_name is None:
                print("Please provide a step name with -name=step_name")
            else:
                tourney_result_list = trace_log_service().get_tourney_results_for_step_name(step_name)
                tourney_result_list = [tourney_result.to_dict() for tourney_result in tourney_result_list]
                json_str = json.dumps(tourney_result_list, indent=4)
                print(json_str)
        elif args.function == 'llmiter':
            file_name = args.file
            model_name = args.name
            lines = args.lines
            result = get_rate_limiter_monitor().graph_model_requests(file_name, model_name, lines)
            print("plot_file_name:"+result)
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(stack_trace)
        print(str(e))
    finally:
        llmonpy_stop()
        exit(0)

if __name__ == "__main__":
    llmonpy_cli()