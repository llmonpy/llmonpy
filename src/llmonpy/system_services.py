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

class SystemServices:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SystemServices, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.config = None
        self.services_to_stop = []
        self.trace_log_service = None

    def set_config(self, config):
        self.config = config
        self.add_service_to_stop(config)

    def set_trace_log_service(self, trace_log_service):
        self.trace_log_service = trace_log_service
        self.add_service_to_stop(trace_log_service)

    def add_service_to_stop(self, service):
        self.services_to_stop.append(service)

    def stop(self):
        stop_in_reverse_order_started = list(reversed(self.services_to_stop))
        for service in stop_in_reverse_order_started:
            service.stop()


def init_system_services():
    SystemServices()

def system_services() -> SystemServices:
    return SystemServices._instance
