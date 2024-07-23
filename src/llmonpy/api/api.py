#  Copyright © 2024 Thomas Edward Burns
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#  documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#  permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#  Software.
#
#  THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#  WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import os

from flask import jsonify, request, send_from_directory

from llmonpy.api.api_app import app
from llmonpy.api.api_config import api_config
from llmonpy.api.api_system_startup import api_system_startup, api_system_stop
from llmonpy.trace_log import trace_log_service

TRACE_ID_PARAM = "trace_id"
STEP_NAME_PARAM = "step_name"
STEP_ID_PARAM = "step_id"

global_static_directory = None

# make default request resolve to index.html, necessary for vuejs router to work well.  Implies no 404 possible
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return send_from_directory(global_static_directory, "index.html")


@app.route('/assets/<path:path>')
def serve_static(path):
    path = "assets/" + path
    return send_from_directory(global_static_directory, path)


@app.route("/qbawa")
def serve_datasets():
    return send_from_directory(global_static_directory, "index.html")


@app.route("/favicon.ico")
def serve_favicon():
    return send_from_directory(global_static_directory, "favicon.ico")


@app.route('/api/hello_world')
def test_json_api():
    result = {"hello": "world"}
    return jsonify(result)


@app.route('/api/get_trace_list')
def get_trace_list():
    result_list = trace_log_service().get_trace_list()
    dict_list = [trace_info.to_dict() for trace_info in result_list]
    return jsonify(dict_list)


@app.route('/api/get_complete_trace_by_id')
def get_complete_trace_by_id():
    trace_id = request.args.get(TRACE_ID_PARAM)
    result = trace_log_service().get_complete_trace_by_id(trace_id)
    return jsonify(result.to_dict())


@app.route('/api/get_tourney_step_name_list')
def get_tourney_step_name_list():
    result_list = trace_log_service().get_tourney_step_name_list()
    return jsonify(result_list)


@app.route('/api/get_tourney_results_for_step_name')
def get_tourney_results_for_step():
    step_name = request.args.get(STEP_NAME_PARAM)
    result_list = trace_log_service().get_tourney_results_for_step_name(step_name)
    dict_list = [trace_info.to_dict() for trace_info in result_list]
    return jsonify(dict_list)


@app.route('/api/get_events_for_step')
def get_events_for_step():
    step_name = request.args.get(STEP_ID_PARAM)
    result_list = trace_log_service().get_events_for_step(step_name)
    dict_list = [trace_info.to_dict() for trace_info in result_list]
    return jsonify(dict_list)


def init_api_directory():
    global global_static_directory
    if global_static_directory is None:
        api_file_path = os.path.abspath(__file__)
        api_dir = os.path.dirname(api_file_path)
        global_static_directory = os.path.join(api_dir, "static")
        print("API Directory: " + global_static_directory)


def run_api():
    init_api_directory()
    api_system_startup()
    try:
        port = api_config().port
        print("Run API on port: " + str(port))
        app.run(port=port)
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    init_api_directory()
    api_system_startup()
    try:
        port = api_config().port
        print("Run API on port: " + str(port))
        app.run(port=port)
    except Exception as e:
        print(str(e))
