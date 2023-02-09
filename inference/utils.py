import datetime
import time


def next_weekday(input):
    # Public holidays are not excluded, only weekends.

    out_days = []
    temp = datetime.datetime.strptime(input[-1], "%Y-%m-%d")
    for _ in range(5):
        next_day = temp + datetime.timedelta(days=1)
        while next_day.isoweekday() in set((6, 7)):
            next_day = next_day + datetime.timedelta(days=1)
        temp = next_day
        out_days.append(temp.strftime("%Y-%m-%d"))

    return out_days


"""
# ------------------------------#
# patch time.time for testing #
# ------------------------------#

class TimeGenerator:
    def __init__(self):
        self.time = 1640995200

    def __call__(self):
        return next(self.generate_next_time())

    def generate_next_time(self):
        self.current_time = self.time
        self.time += 86400
        yield self.current_time


time.time = TimeGenerator()
"""
