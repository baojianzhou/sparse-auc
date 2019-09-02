#!/network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python
from multiprocessing import Pool
import os
import time

work = (["A", 5], ["B", 2], ["C", 1], ["D", 3], ["E", 4], ["F", 5])


def work_log(work_data):
    print(" Process {0} waiting {1} seconds".format(work_data[0], work_data[1]))
    time.sleep(int(work_data[1]))
    print(" Process {0} Finished.".format(work_data[0]))


def pool_handler():
    p = Pool(4)
    p.map(work_log, work)


if __name__ == '__main__':
    print("""USER {user} was granted {cpus} cores and {mem} MB per node on {node}.
The job is current running with job # {job}""".format(
        user=os.getenv("SLURM_JOB_USER"),
        cpus=os.getenv("SLURM_CPUS_PER_TASK"),
        mem=os.getenv("SLURM_MEM_PER_CPU"),
        node=os.getenv("SLURM_NODELIST"),
        job=os.getenv("SLURM_JOB_ID")
    ))
    pool_handler()
