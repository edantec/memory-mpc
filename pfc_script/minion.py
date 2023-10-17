#!/usr/bin/env python3
"""
minion intended to be launched by slurm
"""

import sys
import argparse
import queue
import time

from manager import QueueClient
from work import Problem
from hpp.utils import ServerManager

class Minion(QueueClient):
    def __init__(self, n_tasks: int):
        with ServerManager(server="hpp-manipulation-server"):
            self.n_tasks = n_tasks
            self.problem = Problem()
            super().__init__()

    def run(self):
        for n in range(self.n_tasks):
            try:
                task = self.queue.get(block=False)
            except queue.Empty:
                print("Waiting for next task")
                time.sleep(10)
                continue
            if task[0] == "end_job":
                print("Tasks finished")
                self.queue.put(["end_job"])
                break
            else:
                print(f"start work {n + 1}/{self.n_tasks} on task {task}...")
                wps, timing, targetSample = self.problem.createAndSolveGraph(task, n)
                if wps == None: continue
                self.results.put([wps, timing, targetSample])
            
        # Transmit end-of-task to boss	
        self.results.put([])
        print("Task completed by minion!")


if __name__ == "__main__":
    print(f"{sys.path=}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_tasks", type=int, default=1, nargs="?")

    Minion(**vars(parser.parse_args())).run()
