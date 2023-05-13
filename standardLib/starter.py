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

dt1 = datetime.datetime(2020, 7, 20, 11, 30, 0)
dt2 = datetime.datetime(1990, 3, 10, 12, 0, 0)
dt3 = datetime.datetime(2021, 1, 1, 0, 0, 0)
print(dt1)
print(dt2)
print(dt3)


dt1 = datetime.datetime(2020, 7, 20, 11, 30, 0)
dt2 = datetime.datetime(2021, 2, 20, 10, 25, 0)

dtl = dt2 - dt1
print(dtl)

from datetime import datetime

dt1 = datetime(2021, 4, 20, 11, 30, 0)

print(dt1.strftime("%Y-%m-%d"))
print(dt1.strftime("%d-%m-%Y"))
print(dt1.strftime("%m-%Y"))
print(dt1.strftime("%B-%Y"))
print(dt1.strftime("%d %B, %Y"))
print(dt1.strftime("%Y-%m-%d %H:%M:%S"))
print(dt1.strftime("%m/%d/%y %H:%M:%S"))
print(dt1.strftime("%d(%a) %B %Y"))

date_str_1 = '3 March 1995'
date_str_2 = '3/9/1995'
date_str_3 = '21-07-2021'
dt1 = datetime.strptime(date_str_1, '%d %B %Y')
dt2 = datetime.strptime(date_str_2, '%d/%m/%Y')
dt3 = datetime.strptime(date_str_3, '%d-%m-%Y')

print(dt1)
print(dt2)
print(dt3)

import datetime

today = datetime.date.today()
end_of_year = datetime.date(today.year, 12, 31)
diff = (end_of_year - today).days
print(f'Number of days until the end of the year: {diff}')

# Get the current date and time
now = datetime.datetime.now()

# Get the end of the current year
end_of_year = datetime.datetime(now.year, 12, 31, 23, 59, 59)

# Calculate the time remaining
time_remaining = end_of_year - now

# Output the result
print(f"Until the end of the year: {time_remaining}")

import datetime

now = datetime.datetime.now()
end_of_year = datetime.datetime(now.year + 1, 1, 1)
diff = end_of_year - now
print(f'Until the end of the year: {diff}')

dt1 = datetime.datetime(2020, 1, 1, 0,0,0)

print(dt1 + datetime.timedelta(7))
print(dt1 + datetime.timedelta(30))
print(dt1 + datetime.timedelta(hours=30))
print(dt1 + datetime.timedelta(minutes=15))

import datetime

start_date = datetime.datetime(2020, 1, 1, 0, 0)
end_date = datetime.datetime(2020, 1, 4, 16, 0)
delta = datetime.timedelta(hours=8)

dates = []
while start_date <= end_date:
    dates.append(start_date)
    start_date += delta

for date in dates:
    print(date)

import datetime

rate = 0.04
pv = 1000
daily_rate = rate / 365.0

d1 = datetime.date(2021, 7, 1)
d2 = datetime.date(2021, 12, 31)
duration = d2 - d1

fv = pv * (1 + daily_rate) ** duration.days
print(f'Future value: $ {fv:.2f}')

import string

docs = 'programming in python'
code_map = dict((enumerate(string.ascii_lowercase)))
code_map_inv = {val: key for key, val in code_map.items()}

result = ''
for char in docs:
    if char == ' ':
        result += ' '
        continue
    idx = (code_map_inv[char] + 3) % 26
    result += code_map[idx]

print(result)

from string import Template

names = ['John', 'Paul', 'Lisa', 'Michael']

email = """Hello $name,

Thank you for visiting our website.
Team, XYZ"""

template = Template(email)

for name in names:
    print(template.substitute(name=name))
    print('-' * 35)