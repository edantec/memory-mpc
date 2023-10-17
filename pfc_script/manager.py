#!/usr/bin/env python3

import multiprocessing
import os
from multiprocessing.managers import BaseManager

PORT = 7481
KEY = b"AiZa5Uavcoh3PiajvaeTee5z"  # keep it secret, keep it safe !


class QueueManager(BaseManager):
    """This Manager holds a Queue and waits for clients to use it."""

    pass


class QueueClient:
    """Base class for users of the Queue."""

    def __init__(self):
        QueueManager.register("get_queue")
        QueueManager.register("get_results")
        manager = QueueManager(
            address=(os.environ.get("MANAGER_HOST", "localhost"), PORT), authkey=KEY
        )
        manager.connect()
        self.queue = manager.get_queue()
        self.results = manager.get_results()


if __name__ == "__main__":
    queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    QueueManager.register("get_queue", callable=lambda: queue)
    QueueManager.register("get_results", callable=lambda: result_queue)
    try:
        QueueManager(address=("", PORT), authkey=KEY).get_server().serve_forever()
    finally:
        print()
        print(f"exiting with approximately {queue.qsize()} items left in queue")
