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

from llmonpy.llm_client import init_llm_clients
from llmonpy.system_services import add_service_to_stop

DEFAULT_THREAD_POOL_SIZE = 100
DEFAULT_DATA_DIRECTORY = "data"


def compute_default_data_directory():
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, DEFAULT_DATA_DIRECTORY)
    return data_directory


class LLMonPyConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLMonPyConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self, data_directory=None,
                 thread_pool_size=DEFAULT_THREAD_POOL_SIZE):
        self.thread_pool_size = thread_pool_size
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_pool_size)
        self.data_directory = data_directory if data_directory else compute_default_data_directory()

    def stop(self):
        self.thread_pool.shutdown()

    @staticmethod
    def get_instance():
        return LLMonPyConfig._instance


def init_llmonpy():
    config = LLMonPyConfig()
    add_service_to_stop(config)
    init_llm_clients(data_directory=config.data_directory)
    if os.path.isdir(config.data_directory) is False:
        os.makedirs(config.data_directory)


def llmonpy_config() -> LLMonPyConfig:
    return LLMonPyConfig.get_instance()