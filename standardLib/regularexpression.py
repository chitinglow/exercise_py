import re

text = 'Python 3.8'
print(re.findall(pattern=r'\d', string=text))

print(re.findall(pattern=r'\D', string=text))

code = '0010-000-423'
print(re.findall(pattern=r"[^0-]", string=code))

code = '0010-000-423-22'
print(re.findall(pattern=r"[^-]+", string=code))
print(re.split(pattern='-', string=code))

code = 'PL code: XG-GH-110'
print(re.findall(pattern=r"PL|\d+", string=code))

text = (
    "Please send an email to info@template.com or "
    "sales-info@template.it"
)
print(re.findall(r"[\w\.-]+@[\w\.-]+", text))

text = """Python plays a key role in our production pipeline.
Without it a project the size of Star Wars: Episode II would have been very difficult to pull off."""

print(re.findall(pattern=r"\w+", string=text))

text = """Python plays a key role in our production pipeline.
Without it a project the size of Star Wars: Episode II would have been very difficult to pull off."""

print(re.findall(pattern=r"[A-Z]\w+", string=text))

message = 'For more information, please call: 123-456-789'
print(re.findall(r"\d{3}-\d{3}-\d{3}", message)[0])

text = (
    "Please send an email to info@template.com or "
    "call to: 123-456-789."
)
print(re.sub(r"\d{3}-\d{3}-\d{3}", '***-***-***', text))

from pprint import pprint


data_dict = {
    "users": [
        {
            "userId": 1,
            "firstName": "Krish",
            "lastName": "Lee",
            "emailAddress": "krish.lee@example.com"
        },
        {
            "userId": 2,
            "firstName": "racks",
            "lastName": "jacson",
            "emailAddress": "racks.jacson@example.com"
        },
        {
            "userId": 3,
            "firstName": "denial",
            "lastName": "roast",
            "emailAddress": "denial.roast@example.com"
        }
    ]
}

pprint(data_dict)