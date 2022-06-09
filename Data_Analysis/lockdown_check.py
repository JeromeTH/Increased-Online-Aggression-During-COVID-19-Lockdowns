from datetime import datetime
from dataclasses import dataclass
import csv


@dataclass
class LockdownRecord:
    start: datetime
    end: datetime

    def within(self, date):
        return self.start <= date <= self.end


lockdown_records = {}
default_date = '2000/01/01'

def is_under_lockdown(state, date):
    return lockdown_records[state].within(date)

def lockdown_date(state):
    return lockdown_records[state].start

def open_date(state):
    return lockdown_records[state].end

def no_data(state):

    return lockdown_records[state].start == datetime.strptime(default_date, "%Y/%m/%d")

def never_lockdown(state):
    return lockdown_records[state].start == datetime.strptime(default_date, "%Y/%m/%d")

def ever_lockdown(state):
    return lockdown_records[state].start != datetime.strptime(default_date, "%Y/%m/%d")

def after_lockdown(state, date):
    if never_lockdown(state):
        return False
    return lockdown_records[state].end < date
    
def read_data(filename):
    """
    Reads a csv file that is in format:
    state, start, end
    """
    with open(filename, 'r', encoding='utf8') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # skip header
        for row in reader:
            #print(row)
            try:
                lockdown_records[row[0]] = LockdownRecord(datetime.strptime(row[1], "%Y/%m/%d"), datetime.strptime(row[2], "%Y/%m/%d"))
            except:
                lockdown_records[row[0]] = LockdownRecord(datetime.strptime(default_date, "%Y/%m/%d"), datetime.strptime(default_date, "%Y/%m/%d"))
                


read_data("lockdown.csv")
# uncomment this to test it yourself
# print(is_under_lockdown('California', datetime(2020, 3, 1)))
