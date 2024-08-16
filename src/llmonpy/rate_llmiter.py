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
import threading
from queue import Queue, Empty


class UsedTicket:
    def __init__(self, ticket: int, time_in_seconds: int, token_estimate: int):
        self.ticket = ticket
        self.time_in_seconds = time_in_seconds
        self.token_estimate = token_estimate


class SecondTicketBucket:
    def __init__(self, ticket_count):
        self.ticket_count = ticket_count
        self.used_ticket_list = []

    def get_ticket(self, time_in_seconds, token_estimate):
        result = None
        if self.ticket_count > 0:
            result = UsedTicket(self.ticket_count, time_in_seconds, token_estimate)
            self.used_ticket_list.append(result)
            self.ticket_count -= 1
        return result

    def apply_ramp_profile(self, profile_factor):
        if self.ticket_count > 1:  # don't apply profile to buckets with 1 or 0 tickets
            self.ticket_count = int(self.ticket_count * profile_factor)
            if self.ticket_count < 1:
                self.ticket_count = 1


RAMP_PROFILE = [0.0333, 0.0666, 0.0999, 0.1332, 0.1665, 0.1998, 0.2331, 0.2664, 0.2997, 0.333, 0.3663, 0.3996, 0.4329,
                0.4662, 0.4995, 0.5328, 0.5661, 0.5994, 0.6327, 0.666, 0.6993, 0.7326, 0.7659, 0.7992, 0.8325, 0.8658,
                0.8991, 0.9324, 0.9657, 0.999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


class MinuteTicketBucket:
    def __init__(self, time, minute, second_ticket_count_list:[int] = None, second_bucket_list:[SecondTicketBucket] = None):
        self.time = time
        self.minute = minute
        if second_ticket_count_list is not None:
            self.second_bucket_list = []
            for ticket_count in second_ticket_count_list:
                self.second_bucket_list.append(SecondTicketBucket(ticket_count))
        else:
            self.second_bucket_list = second_bucket_list

    def apply_ramp_profile(self, ramp_profile):
        for index in range(60):
            self.second_bucket_list[index].apply_ramp_profile(ramp_profile[index])

    @staticmethod
    def create_full_speed_bucket(time, minute, ticket_count):
        second_ticket_count_list = []
        if ticket_count < 60:
            # not optimal because ignoring remainder, but this an unimportant edge case
            seconds_per_ticket = int(60 / ticket_count)
            for index in range(60):
                if index % seconds_per_ticket == 0:
                    second_ticket_count_list.append(1)
                else:
                    second_ticket_count_list.append(0)
        else:
            tickets_per_second = ticket_count / 60
            int_tickets_per_second = int(tickets_per_second)
            remainder = tickets_per_second - int_tickets_per_second
            accumulated_remainder = 0
            for index in range(60):
                tickets_in_bucket = int_tickets_per_second
                if accumulated_remainder >= 1:
                    tickets_in_bucket += 1
                    accumulated_remainder = 0
                accumulated_remainder += remainder
                second_ticket_count_list.append(tickets_in_bucket)
        result = MinuteTicketBucket(time, minute, second_ticket_count_list)
        return result




class RateLlmiterMonitor:
    _instance = None

    def __init__(self):
        self.rate_limiter_list = []
        self.timer = threading.Timer(interval=1, function=self.add_tickets)
        self.timer.start()

    def add_rate_limiter(self, rate_limiter):
        self.rate_limiter_list.append(rate_limiter)

    def add_tickets(self):
        for rate_limiter in self.rate_limiter_list:
            rate_limiter.add_tickets()
        self.timer = threading.Timer(interval=1, function=self.add_tickets)
        try:
            self.timer.start()
        except RuntimeError as re:
            pass  # not great...but this happens as we are exiting the program

    @staticmethod
    def get_instance():
        if RateLlmiterMonitor._instance is None:
            RateLlmiterMonitor._instance = RateLlmiterMonitor()
        return RateLlmiterMonitor._instance


class RateLlmiter:
    def __init__(self, request_per_minute, tokens_per_minute):
        self.request_per_minute = request_per_minute
        self.tokens_per_minute = tokens_per_minute
        if request_per_minute < 60:
            self.time_window = 60
            self.intervals_before_refill = 1
            self.request_per_interval = request_per_minute
        else:
            self.time_window = 1
            self.intervals_before_refill = 60
            self.request_per_interval = round(self.request_per_minute / 60)
        self.current_interval = 0
        self.request_limit_queue = Queue()
        self.token_rate_limit_exceeded_queue = Queue()
        self.token_rate_limit_exceeded_count = 0
        self.token_rate_limit_exceeded_lock = threading.Lock()
        RateLlmiterMonitor.get_instance().add_rate_limiter(self)

    def add_tickets(self):
        add_to_request_limit_queue = self.request_per_interval
        add_too_rate_limit_exceeded_queue = 0
        with self.token_rate_limit_exceeded_lock:
            self.current_interval += 1
            if self.current_interval % self.intervals_before_refill != 0 and self.token_rate_limit_exceeded_count > 0:
                self.add_to_request_limit_queue = 0 # if we are not at the end of the interval, don't add any tickets
            elif ((self.current_interval % self.intervals_before_refill) == 0) and self.token_rate_limit_exceeded_count > 0:
                if self.token_rate_limit_exceeded_count < self.request_per_interval:
                    add_too_rate_limit_exceeded_queue = self.token_rate_limit_exceeded_count
                    add_to_request_limit_queue = self.request_per_interval - self.token_rate_limit_exceeded_count
                    self.token_rate_limit_exceeded_count = 0
                else:
                    self.token_rate_limit_exceeded_count -= self.request_per_interval
                    add_too_rate_limit_exceeded_queue = self.request_per_interval
                    add_to_request_limit_queue = 0
        for _ in range(add_too_rate_limit_exceeded_queue):
            self.token_rate_limit_exceeded_queue.put("ticket")
        while not self.request_limit_queue.empty():
            self.request_limit_queue.get_nowait()
        for _ in range(add_to_request_limit_queue):
            self.request_limit_queue.put("ticket")

    def get_ticket(self):
        try:
            result = self.request_limit_queue.get()
        except Empty as empty:
            raise empty
        return result

    def wait_for_ticket_after_rate_limit_exceeded(self):
        with self.token_rate_limit_exceeded_lock:
            self.token_rate_limit_exceeded_count += 1
        try:
            result = self.token_rate_limit_exceeded_queue.get()
        except Empty as empty:
            raise empty
        return result