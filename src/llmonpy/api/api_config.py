#   Copyright © 2024 Thomas Edward Burns
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
#   documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
#   permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
#   Software.
#
#   THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
#   WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#   COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#   OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
from llmonpy.config import LLMonPyConfig, DEFAULT_THREAD_POOL_SIZE
from llmonpy.llm_client import ALL_CLIENT_LIST
from llmonpy.system_services import system_services


class APIConfig(LLMonPyConfig):
    def __init__(self, data_directory=None,
                 client_list=ALL_CLIENT_LIST,
                 thread_pool_size=DEFAULT_THREAD_POOL_SIZE,
                 port=2304):
        super().__init__(data_directory, client_list, thread_pool_size)
        self.port = port


def init_api():
    config = APIConfig()
    system_services().set_config(config)


def api_config() -> APIConfig:
    return system_services().config