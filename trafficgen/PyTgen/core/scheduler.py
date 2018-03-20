'''
Python 3 adaption/modification of the PyTgen scheduler
'''

import datetime
import threading
import heapq
import random
import logging

class scheduler(threading.Thread):
    class job(object):
        def __init__(self, name, action, interval, start, end):
            self.__name = name
            self.__action = action
            self.__interval = interval
            self.__start = start
            self.__end = end
            self.__exec_time = datetime.datetime.now() + datetime.timedelta(seconds = self.__interval[1] * random.random(),
                                                                            minutes = self.__interval[0] * random.random())

        def __call__(self):
            today = datetime.datetime.now()
            start = today.replace(hour = self.__start[0],
                                  minute = self.__start[1],
                                  second = 0, microsecond = 0)
            end = today.replace(hour = self.__end[0],
                                minute = self.__end[1],
                                second = 0, microsecond = 0)

            if start <= self.__exec_time < end:
                # enqueue job for "random() * 2 * interval" 
                # in average the job will run every interval but differing randomly
                self.__exec_time += datetime.timedelta(seconds = self.__interval[1] * random.random() * 2,
                                                       minutes = self.__interval[0] * random.random() * 2)

                if self.__exec_time < datetime.datetime.now():
                    logging.getLogger("scheduler").warning('scheduler is overloaded!')

                return self.__action

            else:
                # enqueue the job until next start time
                if self.__exec_time < start and self.__exec_time.day == start.day:
                    self.__exec_time = start + datetime.timedelta(seconds = 1)
                else:
                    self.__exec_time = start + datetime.timedelta(days = 1)

                logging.getLogger("scheduler").info("enqueueing %s until %s",
                                                    self.__name, self.__exec_time)
                return False


        def __lt__(self, other):
            try:
                if type(other) == scheduler.job:
                    return self.__exec_time < other.__exec_time

                elif type(other) == datetime.datetime:
                    return self.__exec_time < other
                else:
                    raise Exception()
            except Exception as e:
                raise 
            
        def __sub__(self, other):
            try:
                if type(other) == scheduler.job:
                    return self.__exec_time - other.__exec_time

                elif type(other) == datetime.datetime:
                    return self.__exec_time - other
                else:
                    raise Exception()
            except Exception as e:
                raise

    def __init__(self, jobs, runner):
        threading.Thread.__init__(self)
        self.setName('scheduler')

        self.__runner = runner
        self.__jobs = jobs
        heapq.heapify(self.__jobs)

        self.__running = False
        self.__signal = threading.Condition()

    def run(self):
        self.__running = True
        while self.__running:
            self.__signal.acquire()
            if not self.__jobs:
                self.__signal.wait()

            else:
                now = datetime.datetime.now()
                while (self.__jobs[0] < now):
                    job = heapq.heappop(self.__jobs)

                    action = job()
                    if action is not False:
                        self.__runner(action)

                    heapq.heappush(self.__jobs, job)

                logging.getLogger("scheduler").debug("Sleeping %s seconds", (self.__jobs[0] - now))
                self.__signal.wait((self.__jobs[0] - now).total_seconds())

            self.__signal.release()

    def stop(self):
        self.__running = False

        self.__signal.acquire()
        self.__signal.notify_all()
        self.__signal.release()

    def set_jobs(self, jobs):
        self.__signal.acquire()

        self.__jobs = jobs
        heapq.heapify(self.__jobs)

        self.__signal.notify_all()
        self.__signal.release()