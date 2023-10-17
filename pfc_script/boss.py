#!/usr/bin/env python3

import random

from manager import QueueClient
import numpy as np
import queue
import pickle
import time

def sampleRange(samples_range):
    samples_vector = []
    for sample_range in samples_range:
        samples_vector.append(np.random.uniform(sample_range[0],sample_range[1]))
    return samples_vector

class Boss(QueueClient):
    def run(self):
        x_goals = np.array([0.6,0.9])
        y_goals = np.array([0.0,0.6])
        z_goals = np.array([0.8,1.2])
        target_ranges = [x_goals, y_goals, z_goals]
        for _ in range(10):
            task = sampleRange(target_ranges)
            print("new task:", task)
            self.queue.put(task)
        self.queue.put(["end_job"])

        # Create data storage
        data_dict = {}
        wps_data = []
        timings_data = []
        target_data = []

        # Gather minions' results
        minions_end = 0
        tasks_received = 0
        while minions_end < 11:
            try:
                result = self.results.get(block=False)
            except queue.Empty:
                print("Queue empty, wait for results to arrive")
                time.sleep(20)
                continue
            if result == []:
                minions_end += 1
                print(f"Minion {minions_end} has completed his job!")
            else:
                tasks_received += 1
                wps_data.append(result[0])
                timings_data.append(result[1])
                target_data.append(result[2])
        
        data_dict['waypoints'] = np.array(wps_data)
        data_dict['timings'] = np.array(timings_data)
        data_dict['targets'] = np.array(target_data)

        with open('reduced_trajectories.pkl', 'wb') as f: 
            pickle.dump(data_dict, f)
        print("Data stored in reduced_trajectories.pkl")


if __name__ == "__main__":
    Boss().run()
