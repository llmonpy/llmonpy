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
import concurrent
import os

from llmonpy.llm_client import add_llm_clients, ALL_CLIENT_LIST
from llmonpy.system_services import system_services

DEFAULT_THREAD_POOL_SIZE = 50
DEFAULT_DATA_DIRECTORY = "data"


def compute_default_data_directory():
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, DEFAULT_DATA_DIRECTORY)
    return data_directory


class LLMonPyConfig:
    def __init__(self, data_directory=None,
                 client_list=ALL_CLIENT_LIST,
                 thread_pool_size=DEFAULT_THREAD_POOL_SIZE):
        self.client_list = add_llm_clients(client_list)
        self.thread_pool_size = thread_pool_size
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_pool_size)
        self.data_directory = data_directory if data_directory else compute_default_data_directory()

    def stop(self):
        self.thread_pool.shutdown()


def init_llmonpy():
    config = LLMonPyConfig()
    system_services().set_config(config)
    if os.path.isdir(config.data_directory) is False:
        os.makedirs(config.data_directory)


def llmonpy_config() -> LLMonPyConfig:
    return system_services().config