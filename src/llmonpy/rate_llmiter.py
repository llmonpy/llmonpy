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
import copy
import datetime
import threading
import time
from queue import Queue, Empty

MIN_TEST_IF_SERVICE_RESUMED_INTERVAL = 10
MAX_TEST_IF_SERVICE_RESUMED_INTERVAL = 120
INTERVAL_BACKOFF_RATE = 2.5

class RateLimitedService:
    def test_if_blocked(self) -> bool:
        raise NotImplementedError


class RateLimitedEvent:
    def __init__(self, second_bucket_ticket_was_issued: int, second_bucket_ticket_was_limited: int,
                 reissued_second_bucket_id: int = None):
        self.second_bucket_ticket_was_issued = second_bucket_ticket_was_issued
        self.second_bucket_ticket_was_limited = second_bucket_ticket_was_limited
        self.reissued_second_bucket_id = reissued_second_bucket_id

    def is_waiting(self):
        result = self.reissued_second_bucket_id is None
        return result


class RateLlmiterTicket:
    def __init__(self, request_id:int, initial_request_second_bucket_id: int, issued_ticket: int= None,
                 issued_second_bucket_id: int = None, rate_limit_event_list: [RateLimitedEvent] = None):
        self.request_id = request_id
        self.initial_request_second_bucket_id = initial_request_second_bucket_id
        self.issued_ticket = issued_ticket
        self.issued_second_bucket_id = issued_second_bucket_id
        self.rate_limit_event_list = rate_limit_event_list if rate_limit_event_list is not None else []

    def has_issued_ticket(self):
        result = self.issued_ticket is not None
        return result

    def record_issued_ticket(self, issued_ticket, second_bucket_id):
        self.issued_ticket = issued_ticket
        self.issued_second_bucket_id = second_bucket_id

    def add_rate_limited_event(self, second_bucket_id):
        rate_limited_event = RateLimitedEvent(self.issued_second_bucket_id, second_bucket_id)
        self.rate_limit_event_list.append(rate_limited_event)

    def resolve_rate_limited_event(self, second_bucket_id):
        self.rate_limit_event_list[-1].reissued_second_bucket_id = second_bucket_id


class WaitingTicket:
    def __init__(self, ticket: RateLlmiterTicket):
        self.ticket = ticket
        self.event = threading.Event()

    def wait(self):
        self.event.wait()

    def resume_request(self):
        self.event.set()
        return self.ticket


class SecondTicketBucket:
    def __init__(self, second_bucket_id: int, ticket_count: int = 0, issued_ticket_count: int = 0,
                 issued_ticket_list: [RateLlmiterTicket] = None,
                 overflow_request_list: [RateLlmiterTicket] = None, rate_limited_request_list: [RateLlmiterTicket] = None):
        self.second_bucket_id = second_bucket_id
        self.ticket_count = ticket_count
        self.issued_ticket_count = issued_ticket_count
        self.issued_ticket_list: [RateLlmiterTicket] = issued_ticket_list if issued_ticket_list is not None else []
        # requests that could not be satisfied
        self.overflow_request_list: [RateLlmiterTicket] = overflow_request_list if overflow_request_list is not None else []
        # requests that had tickets issued, but generated rate limit exceptions
        self.rate_limited_request_list: [RateLlmiterTicket] = rate_limited_request_list if rate_limited_request_list is not None else []

    def get_ticket(self, request_id: int):
        result = RateLlmiterTicket(request_id, self.second_bucket_id)
        self.process_ticket_request(result)
        if result.has_issued_ticket() is False:
            self.overflow_request_list.append(result)
        return result

    def process_ticket_request(self, ticket):
        self.issue_ticket(ticket)
        return ticket

    def issue_ticket(self, ticket: RateLlmiterTicket):
        result = False
        if self.issued_ticket_count < self.ticket_count:
            self.issued_ticket_count += 1
            ticket.record_issued_ticket(self.issued_ticket_count, self.second_bucket_id)
            self.issued_ticket_list.append(ticket)
            result = True
        return result

    def add_rate_limited_request(self, ticket: RateLlmiterTicket):
        self.ticket_count = 0
        ticket.add_rate_limited_event(self.second_bucket_id)
        self.rate_limited_request_list.append(ticket)

    def set_ticket_count(self, max_ticket_count, min_ticket_count, prior_bucket_issued_tickets, ticket_count_delta):
        if max_ticket_count == prior_bucket_issued_tickets:
            self.ticket_count = max_ticket_count
        else:
            self.ticket_count = prior_bucket_issued_tickets + ticket_count_delta
        self.ticket_count = min(self.ticket_count, max_ticket_count)
        self.ticket_count = max(self.ticket_count, min_ticket_count)

    def transfer_tickets(self, unsatisfied_request_list: [RateLlmiterTicket],
                         rate_limited_request_list: [RateLlmiterTicket]) -> [RateLlmiterTicket]:
        released_ticket_list = []
        # requests that had tickets issued, but generated rate limit exceptions are reissued first
        for ticket in rate_limited_request_list:
            ticket = copy.deepcopy(ticket)
            if self.process_ticket_request(ticket).has_issued_ticket() is False:
                self.rate_limited_request_list.append(ticket)
            else:
                ticket.resolve_rate_limited_event(self.second_bucket_id)
                released_ticket_list.append(ticket)
        for ticket in unsatisfied_request_list:
            ticket = copy.deepcopy(ticket)
            if self.process_ticket_request(ticket).has_issued_ticket() is False:
                self.overflow_request_list.append(ticket)
            else:
                released_ticket_list.append(ticket)
        return released_ticket_list

    def get_issued_ticket_count(self):
        return self.issued_ticket_count


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
                self.second_bucket_list.append(SecondTicketBucket(next_start_time))
                next_start_time += 1
            self.second_bucket_list[0].ticket_count = first_bucket_ticket_count
        else:
            self.second_bucket_list = second_bucket_list
        self.current_second_bucket_index = 0

    def get_last_bucket_ticket_used_count(self):
        result = self.second_bucket_list[self.current_second_bucket_index].get_issued_ticket_count()
        return result

    def get_ticket(self, request_id) -> RateLlmiterTicket:
        result = self.second_bucket_list[self.current_second_bucket_index].get_ticket(request_id)
        return result

    def add_rate_limited_request(self, ticket: RateLlmiterTicket):
        self.second_bucket_list[self.current_second_bucket_index].add_rate_limited_request(ticket)

    def advance_second_bucket(self, set_ticket_count=True):
        expiring_second_bucket = self.second_bucket_list[self.current_second_bucket_index]
        self.current_second_bucket_index += 1
        self.current_second_bucket_index = min(self.current_second_bucket_index, 59)
        if set_ticket_count:
            self.second_bucket_list[self.current_second_bucket_index].set_ticket_count(self.max_tickets_per_second,
                                                                                   self.start_ramp_ticket_count,
                                                                                   expiring_second_bucket.issued_ticket_count,
                                                                                   self.ramp_ticket_count_delta)

    def release_tickets(self) -> [RateLlmiterTicket]:
        last_second_bucket = self.second_bucket_list[self.current_second_bucket_index - 1]
        result = self.second_bucket_list[self.current_second_bucket_index].transfer_tickets(last_second_bucket.overflow_request_list,
                                                        last_second_bucket.rate_limited_request_list)
        return result

    def transfer_tickets(self, last_minute_bucket) -> [RateLlmiterTicket]:
        if last_minute_bucket is None:
            return []
        else:
            last_used_second_bucket = last_minute_bucket.second_bucket_list[last_minute_bucket.current_second_bucket_index]
            result = self.second_bucket_list[0].transfer_tickets(last_used_second_bucket.overflow_request_list,
                                                        last_used_second_bucket.rate_limited_request_list)
            return result


class RateLlmiterMonitor:
    _instance = None

    def __init__(self):
        self.rate_limiter_list = []
        self.second_bucket_index = 0
        self.start_time_in_seconds = int(time.time())
        self.start_iso_date_string = datetime.datetime.fromtimestamp(self.start_time_in_seconds).isoformat()
        self.timer = threading.Timer(interval=1, function=self.add_tickets)
        self.timer.start()

    def add_rate_limiter(self, rate_limiter):
        rate_limiter.refresh_minute_bucket(self.start_time_in_seconds, self.start_iso_date_string)
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


# This class is used to organize the data that needs to be locked before access
class BucketRateLimiterLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.next_request_index = 0
        self.current_minute_bucket = None
        self.is_paused_by_rate_limit_exception = False
        self.waiting_ticket_dict = {}
        self.test_if_service_resumed_timer = None
        self.service_test_interval = MIN_TEST_IF_SERVICE_RESUMED_INTERVAL


class BucketRateLimiter:
    def __init__(self, request_per_minute, tokens_per_minute, rate_limited_service: RateLimitedService = None):
        self.request_per_minute = request_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.rate_limited_service = rate_limited_service
        if request_per_minute < 60: # this isn't optimal, but don't really care about this use case
            self.start_ramp_ticket_count = 1
            self.max_tickets_per_second = 1
        else:
            self.max_tickets_per_second = int(request_per_minute / 60)
            self.start_ramp_ticket_count = max(int((request_per_minute / 60) * 0.25), 1)
            self.ramp_ticket_count_delta = max(int((request_per_minute / 60) * 0.10), 1)
            print(f"max_tickets_per_second: {self.max_tickets_per_second}, start_ramp_ticket_count: {self.start_ramp_ticket_count}, ramp_ticket_count_delta: {self.ramp_ticket_count_delta}")
        self.thread_safe_data = BucketRateLimiterLock()
        RateLlmiterMonitor.get_instance().add_rate_limiter(self)

    def set_rate_limited_service(self, rate_limited_service: RateLimitedService):
        if rate_limited_service is not None:  # only set once for APIs that use one rate limiter for many models
            self.rate_limited_service = rate_limited_service

    def wait(self, ticket):
        waiting_ticket = WaitingTicket(ticket)
        with self.thread_safe_data.lock:
            self.thread_safe_data.waiting_ticket_dict[ticket.request_id] = waiting_ticket
        waiting_ticket.wait()
        with self.thread_safe_data.lock:
            del self.thread_safe_data.waiting_ticket_dict[ticket.request_id]
        return waiting_ticket.ticket

    def resume_request(self, ticket):
        with self.thread_safe_data.lock:
            waiting_ticket = self.thread_safe_data.waiting_ticket_dict[ticket.request_id]
            waiting_ticket.ticket = ticket # update ticket with issued ticket
        waiting_ticket.resume_request()

    def refresh_minute_bucket(self, time_in_seconds, iso_date_string):
        with self.thread_safe_data.lock:
            last_minute_bucket = self.thread_safe_data.current_minute_bucket
            if self.thread_safe_data.is_paused_by_rate_limit_exception:
                first_bucket_ticket_count = 0
            else:
                first_bucket_ticket_count = last_minute_bucket.get_last_bucket_ticket_used_count() if last_minute_bucket is not None else self.start_ramp_ticket_count
                first_bucket_ticket_count = max(first_bucket_ticket_count, self.start_ramp_ticket_count)
            self.thread_safe_data.current_minute_bucket = MinuteTicketBucket(time_in_seconds, iso_date_string, self.max_tickets_per_second,
                                                            self.start_ramp_ticket_count, first_bucket_ticket_count,
                                                            self.ramp_ticket_count_delta)
            released_ticket_list = self.thread_safe_data.current_minute_bucket.transfer_tickets(last_minute_bucket)
        for ticket in released_ticket_list:
            self.resume_request(ticket)
        return last_minute_bucket

    def release_tickets(self):
        with self.thread_safe_data.lock:
            if self.thread_safe_data.is_paused_by_rate_limit_exception:
                self.thread_safe_data.current_minute_bucket.advance_second_bucket(False)
            else:
                self.thread_safe_data.current_minute_bucket.advance_second_bucket(True)
            released_ticket_list = self.thread_safe_data.current_minute_bucket.release_tickets()
        for ticket in released_ticket_list:
            self.resume_request(ticket)

    def get_ticket(self):
        with self.thread_safe_data.lock:
            request_id = self.thread_safe_data.next_request_index
            self.thread_safe_data.next_request_index += 1
            ticket = self.thread_safe_data.current_minute_bucket.get_ticket(request_id)
        if ticket.has_issued_ticket() is False:
            ticket = self.wait(ticket)
        print(f"ticket: {ticket.request_id}")
        return ticket

    def wait_for_ticket_after_rate_limit_exceeded(self, ticket):
        self.return_ticket(ticket)
        with self.thread_safe_data.lock:
            start_service_resumed_timer = self.thread_safe_data.is_paused_by_rate_limit_exception is False
            self.thread_safe_data.is_paused_by_rate_limit_exception = True
            self.thread_safe_data.current_minute_bucket.add_rate_limited_request(ticket)
        if start_service_resumed_timer:
            self.start_test_if_service_resumed_timer()
        print(f"rate limited ticket: {ticket.request_id}")
        result = self.wait(ticket)
        return result

    def return_ticket(self, ticket):
        pass

    def start_test_if_service_resumed_timer(self):
        # this isn't thread safe, but it will only be called by one thread
        current_interval = self.thread_safe_data.service_test_interval
        timer = threading.Timer(interval=current_interval, function=self.test_if_service_resumed)
        self.thread_safe_data.test_if_service_resumed_timer = timer
        timer.start()

    def test_if_service_resumed(self):
        current_time = int(time.time())
        print("testing if blocked at: ", current_time)
        blocked = self.rate_limited_service.test_if_blocked()
        if blocked:
            self.thread_safe_data.service_test_interval = min(
                self.thread_safe_data.service_test_interval * INTERVAL_BACKOFF_RATE,
                MAX_TEST_IF_SERVICE_RESUMED_INTERVAL)
            self.start_test_if_service_resumed_timer()
        else:
            with self.thread_safe_data.lock:
                self.thread_safe_data.is_paused_by_rate_limit_exception = False
                self.thread_safe_data.test_if_service_resumed_timer = None
                self.thread_safe_data.service_test_delay = MIN_TEST_IF_SERVICE_RESUMED_INTERVAL
                # tickets released in the next second bucket


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