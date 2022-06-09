import datetime
timeline = []
x = datetime.datetime(2020, 3, 29)
x = x + datetime.timedelta(days=10)
end_date = datetime.datetime(2020, 9,  1)
next_date = x + datetime.timedelta(days= 100)

def make_string(x):
    return str(x)[:10]
def get_date(s):
    #2020-04-14
    year = int(s[:4])
    month = int(s[5:7])
    day = int(s[8:10])
    res = datetime.datetime(year, month, day)
    return res

def adday(x, d):
    x = x + datetime.timedelta(days = d)
    return x

