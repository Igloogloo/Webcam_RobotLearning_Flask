import time

class Stopwatch():
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def duration(self):
        if self.start_time is None:
            print("Call start() before duration.")
            return 0
        return time.time() - self.start_time

    def restart(self):
        self.start_time = time.time()