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
import datetime
import threading
import time
from queue import Queue, Empty


class RateLimitedService:
    def test_if_blocked(self):
        raise NotImplementedError


class UsedTicket:
    def __init__(self, ticket_index:int, request_time:int, ticket: int, time_in_seconds: int,
                 input_token_count_estimate: int):
        self.ticket = ticket
        self.time_in_seconds = time_in_seconds
        self.input_token_count_estimate = input_token_count_estimate
        self.granted_time_in_seconds = time_in_seconds




class SecondTicketBucket:
    def __init__(self, start_time_in_seconds, ticket_count):
        self.start_time_in_seconds = start_time_in_seconds
        self.ticket_count = ticket_count
        self.used_ticket_list = []

    def get_ticket(self, time_in_seconds, token_estimate):
        result = None
        if self.ticket_count > 0:
            result = UsedTicket(self.ticket_count, time_in_seconds, token_estimate)
            self.used_ticket_list.append(result)
            self.ticket_count -= 1
        return result

    def get_used_ticket_count(self):
        result = len(self.used_ticket_list)
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
    def __init__(self, start_time_in_seconds, iso_date_string, max_tickets_per_second, start_ramp_ticket_count,
                first_bucket_ticket_count, ramp_ticket_count_delta,
                 second_bucket_list: [SecondTicketBucket] = None):
        self.iso_date_string = iso_date_string
        self.start_time_in_seconds = start_time_in_seconds
        self.max_tickets_per_second = max_tickets_per_second
        self.start_ramp_ticket_count = start_ramp_ticket_count
        self.ramp_ticket_count_delta = ramp_ticket_count_delta
        if second_bucket_list is None:
            self.second_bucket_list = []
            next_start_time = start_time_in_seconds
            for index in range(60):
                self.second_bucket_list.append(SecondTicketBucket(next_start_time, first_bucket_ticket_count))
                next_start_time += 1
        else:
            self.second_bucket_list = second_bucket_list

    def get_last_bucket_ticket_used_count(self):
        result = self.second_bucket_list[-1].get_used_ticket_count()
        return result

    def create_next_minute_bucket(self, iso_date_string, time_in_seconds):
        next_start_time = self.start_time_in_seconds + 60
        next_minute_bucket = MinuteTicketBucket(iso_date_string, next_start_time, second_bucket_list=self.second_bucket_list)
        return next_minute_bucket

    def get_ticket(self, second_bucket_index, ticket_index):
        time_in_seconds = time.time()
        if self.total_ticket_count > 0:
            result = self.second_bucket_list[second_bucket_index].get_ticket(time_in_seconds, token_estimate)
            if result is not None:
                self.used_ticket_count += 1
        return result

    def create_next_minute_bucket(self, iso_date_string, time_in_seconds):
        next_start_time = self.start_time_in_seconds + 60
        next_minute_bucket = MinuteTicketBucket(iso_date_string, next_start_time, second_bucket_list=self.second_bucket_list)
        return next_minute_bucket

    def get_ticket(self, second_bucket_index, ticket_index):
        time_in_seconds = time.time()
        if self.total_ticket_count > 0:
            result = self.second_bucket_list[second_bucket_index].get_ticket(time_in_seconds, token_estimate)
            if result is not None:
                self.used_ticket_count += 1
        return result




class RateLlmiterMonitor:
    _instance = None

    def __init__(self):
        self.rate_limiter_list = []
        self.second_bucket_index = 0
        self.timer = threading.Timer(interval=1, function=self.add_tickets)
        self.timer.start()

    def add_rate_limiter(self, rate_limiter):
        self.rate_limiter_list.append(rate_limiter)

    def add_tickets(self):
        self.second_bucket_index = self.second_bucket_index % 60
        time_in_seconds = int(time.time())
        iso_date_string = datetime.datetime.fromtimestamp(time_in_seconds).isoformat()
        buckets_to_log = []
        for rate_limiter in self.rate_limiter_list:
            if self.second_bucket_index == 0:
                last_bucket = rate_limiter.refresh_minute_bucket(time_in_seconds, iso_date_string)
                if last_bucket is not None:
                    buckets_to_log.append(last_bucket)
            else:
                rate_limiter.release_tickets()
        self.second_bucket_index += 1
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


class BucketRateLimiter:
    def __init__(self, request_per_minute, tokens_per_minute, rate_limited_service: RateLimitedService):
        self.request_per_minute = request_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.rate_limited_service = rate_limited_service
        self.next_request_index = 0
        self.current_minute_bucket = None
        if request_per_minute < 60:
            self.start_ramp_ticket_count = 1
            self.max_tickets_per_second = 1
        else:
            self.max_tickets_per_second = int(request_per_minute / 60)
            self.start_ramp_second_ticket_count = int((request_per_minute / 60) * 0.25)
            self.ramp_ticket_count_delta = int((request_per_minute / 60) * 0.10)
            if self.ramp_ticket_count_delta < 1:
                self.ramp_ticket_count_delta = 1
            if self.start_ramp_second_ticket_count < 1:
                self.start_ramp_second_ticket_count = 1
        self.bucket_lock = threading.Lock()
        RateLlmiterMonitor.get_instance().add_rate_limiter(self)

    def refresh_minute_bucket(self, time_in_seconds, iso_date_string):
        last_minute_bucket = self.current_minute_bucket
        first_bucket_ticket_count = last_minute_bucket.get_last_bucket_ticket_used_count() if last_minute_bucket is not None else self.start_ramp_ticket_count
        if first_bucket_ticket_count < self.start_ramp_ticket_count:
            first_bucket_ticket_count = self.start_ramp_ticket_count
        self.current_minute_bucket = MinuteTicketBucket(time_in_seconds, iso_date_string, self.max_tickets_per_second,
                                                        self.start_ramp_ticket_count, first_bucket_ticket_count,
                                                        self.ramp_ticket_count_delta)
        return last_minute_bucket

    def release_tickets(self):
        with self.bucket_lock:
            released_ticket_list = self.current_minute_bucket.release_tickets()
        for ticket in released_ticket_list:
            release tickets

    def get_ticket(self):
        with self.bucket_lock:
            request_id = self.next_request_index
            self.next_request_index += 1
            ticket = self.current_minute_bucket.get_ticket(request_id)
        if ticket.is_rate_limited():
            ticket.wait()
        return ticket

    def wait_for_ticket_after_rate_limit_exceeded(self):
        should pass in the used ticket for original token
        deletes tickets current & future seconds
        sets rate limited to true
        rate limited requests stored in list for no longer rate limited


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