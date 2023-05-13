from collections import Counter

target = ['yes', 'no', 'no', None, 'yes', 'yes', 'no', 'yes']
cnt = Counter(target)
print(cnt)

poll_1 = {'yes': 140, 'no': 120, None: 12}
poll_2 = {'yes': 113, 'no': 132, None: 9}

cnt_1 = Counter(poll_1)
cnt_2 = Counter(poll_2)

cnt_total = cnt_1 + cnt_2
print(cnt_total)

# With filter
poll_1 = {'yes': 140, 'no': 120, None: 12}
poll_2 = {'yes': 113, 'no': 132, None: 9}
poll_3 = {'yes': 16, 'no': 14}

cnt_1 = Counter(poll_1)
cnt_2 = Counter(poll_2)
cnt_3 = Counter(poll_3)

cnt_total = cnt_1 + cnt_2 + cnt_3
print(cnt_total['yes'])

from collections import Counter

text = 'python programming - introduction'

cnt = Counter(text)
print(cnt)

print(Counter(text).most_common(3))

from collections import Counter
import re

text = """"Python is fast enough for our site and allows us to produce maintainable features in record times,
with a minimum of developers," said Cuong Do, Software Architect, YouTube.com.

"Python plays a key role in our production pipeline. Without it a project the size of Star Wars:
Episode II would have been very difficult to pull off. From crowd rendering to batch processing to compositing,
Python binds all things together," said Tommy Burnette, Senior Technical Director, Industrial Light & Magic."""

tokens = re.findall(r"\w+", text)
tokens = [token.lower() for token in tokens]
cnt = Counter(tokens)
print(cnt.most_common(3))

from collections import Counter

fnames = [
    '001_image.png',
    '002_image.png',
    '003_image.jpg',
    '004_image.png',
    '005_image.png',
    '006_image.png',
    '007_image.jpg',
    '008_image.png',
    '009_image.jpg',
    '010_image.jpg',
    '011_image.jpg',
    '012_image.png',
    '013_image.jpg',
    '014_image.jpg',
    '015_image.jpg',
    '016_image.png',
    '017_image.jpg',
    '018_image.jpg',
    '019_image.png',
    '020_image.png',
    '021_image.jpg',
    '022_image.jpg',
    '023_image.png',
    '024_image.png',
    '025_image.jpg',
    '026_image.png',
    '027_image.png',
    '028_image.jpg',
    '029_image.png',
    '030_image.png',
]

extensions = [fname.split('.')[1] for fname in fnames]
cnt = Counter(extensions)
print(cnt)

from collections import ChainMap

stocks_1 = {'CDR': 400, 'PLW': 490}
stocks_2 = {'PLW': 500, 'TEN': 550, 'BBT': 30}

stocks = ChainMap(stocks_1, stocks_2)
print(stocks)

from collections import ChainMap

stocks_1 = {'CDR': 400, 'PLW': 490}
stocks_2 = {'PLW': 500, 'TEN': 550, 'BBT': 30}

stocks = ChainMap(stocks_1, stocks_2)
print(stocks['PLW'])

from collections import ChainMap

techs = {'Apple': 370, 'Samsung': 1100, 'Microsoft': 201}
finance = {'Citigroup': 51, 'Allianz': 180}
gaming = {'Sony': 76, 'Nintendo': 470, 'EA': 135}

stocks = ChainMap(techs, finance, gaming)
print(stocks['Samsung'])

techs['Microsoft'] = 210
print(stocks['Microsoft'])
stocks = ChainMap(techs, finance, gaming)
print(sorted(list(stocks.keys())))

default_settings = {'mode': 'eco', 'power_level': 7}
user_settings = {'mode': 'sport', 'power_level': 10}

settings = ChainMap(user_settings, default_settings)
print(settings['mode'])

default_settings = {'mode': 'eco', 'power_level': 7}
user_settings = {}

settings = ChainMap(user_settings, default_settings)
print(settings['mode'])