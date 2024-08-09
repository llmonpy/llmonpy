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
from llmonpy.system_services import add_service_to_stop


class APIConfig(LLMonPyConfig):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(APIConfig, cls).__new__(cls)
        return cls._instance

    def __init__(self, data_directory=None,
                 thread_pool_size=DEFAULT_THREAD_POOL_SIZE,
                 port=2304):
        super().__init__(data_directory, thread_pool_size)
        self.port = port

    @staticmethod
    def get_instance():
        return APIConfig._instance

def init_api():
    config = APIConfig()
    add_service_to_stop(config)


def api_config() -> APIConfig:
    result = APIConfig.get_instance()
    return result