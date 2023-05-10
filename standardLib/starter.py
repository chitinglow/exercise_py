import sys

print(sys.version.split(' ')[0])

import datetime

d1 = datetime.date(2021, 1, 1)
d2 = datetime.date(2021, 7, 31)
d3 = datetime.date(1990, 5, 7)

print(d1)
print(d2)
print(d3)

t1 = datetime.time(12, 00, 0)
t2 = datetime.time(6, 30, 0)
t3 = datetime.time(9, 15, 0)

print(t1, t2, t3)