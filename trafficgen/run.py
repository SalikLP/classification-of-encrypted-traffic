'''
Python 3 adaption/modification of the PyTgen run.py
'''

import logging
import signal
import platform

import core.runner
import core.scheduler
import config

from core.generator import *


def create_jobs():
    logging.getLogger('main').info('Creating jobs')

    jobs = []
    for next_job in Conf.jobdef:
        logging.getLogger('main').info('creating %s', next_job)

        job = core.scheduler.job(name = next_job[0],
                                 action = eval(next_job[0])(next_job[2]),
                                 interval = next_job[1][2],
                                 start = next_job[1][0],
                                 end = next_job[1][1])
        jobs.append(job)

    return jobs

if __name__ == '__main__':
    # set hostbased parameters
    hostname = platform.node()
    
    log_file = 'trafficgen/logs/' + hostname + '.log'
    config_file = "config.py"
    # file = open(log_file, 'a+')
    # file.close()
    # load the hostbased configuration file
    # _Conf = __import__(config_file, globals(), locals(), ['Conf'], -1)
    # Conf = _Conf.Conf
    Conf = config.Conf

    # start logger
    logging.basicConfig(level = Conf.loglevel,
                        format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt = '%Y-%m-%d %H:%M:%S',
                        filename = log_file)

    logging.getLogger('main').info('Configuration %s loaded', config_file)

    # start runner, create jobs, start scheduling
    runner = core.runner(maxthreads = Conf.maxthreads)

    jobs = create_jobs()

    scheduler = core.scheduler(jobs = jobs,
                               runner = runner)

    # Stop scheduler on exit
    def signal_int(signal, frame):
        logging.getLogger('main').info('Stopping scheduler')
        scheduler.stop()

    signal.signal(signal.SIGINT, signal_int)

    # Run the scheduler
    logging.getLogger('main').info('Starting scheduler')
    scheduler.start()
    scheduler.join(timeout=None)

    # Stop the runner
    runner.stop()