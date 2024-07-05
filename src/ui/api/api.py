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

from flask import jsonify, request

from api_app import app
from api_config import api_config
from api_system_startup import api_system_startup, api_system_stop
from trace_log import trace_log_service

TRACE_ID_PARAM = "trace_id"
STEP_NAME_PARAM = "step_name"
STEP_ID_PARAM = "step_id"

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


if __name__ == "__main__":
    api_system_startup()
    try:
        port = api_config().port
        print("Run API on port: " + str(port))
        app.run(port=port)
    except Exception as e:
        print(str(e))
