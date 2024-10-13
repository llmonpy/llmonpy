import os
import sys
import traceback

from click import prompt

from llmonpy.system_startup import llmonpy_start, llmonpy_stop

if __name__ == "__main__":
    llmonpy_start()
    try:
        request_name = sys.argv[1] + ".txt"
        request_file_path = os.path.join(os.getcwd(), "src/experiments/title_write/" + request_name)
        with open(request_file_path, "r") as file:
            request = file.read()
        finish prompt
        create tourament for prompt
        write out qbawa for tournament (trace_id)
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(stack_trace)
        print("Error:", str(e))
    finally:
        llmonpy_stop()
        exit(0)