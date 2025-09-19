import os
from datetime import datetime

class Logging:
    def __init__(self, log_path):
        self.f = open(log_path, "w")
        self.format = "[{time}][{level}] {message}"
        self.date_format = "%H:%M:%S"

    def log_and_write(level, message):
        log_message = self.format.format(time = datetime.now().strftime(self.date_format),
                                         level = level,
                                         message = message) 
        print(log_message)
        self.f.write(log_message + "\n")

    def close(self):
        self.f.close()