#-*- coding:utf-8 -*-
import requests
import numpy as np
import json

addr = 'http://localhost:5000'
test_url = addr + '/api/test'
content_type = 'application/json'
headers = {'content-type': content_type}

data1 = {'text': 'how many animals do you know?',
        'model': 1,
        'language': 'english',
        'speed': 1,
        'tone': 1,
        'volume': 1}

data2 = {'text': 'how many animals do you know?',
        'model': 2,
        'language': 'english',
        'speed': 1,
        'tone': 1,
        'volume': 1}

data3 = {'text': 'how many animals do you know?',
        'model': 3,
        'language': 'english',
        'speed': 1,
        'tone': 1,
        'volume': 1}

data4 = {'text': '关于西藏的传说有很多.',
        'model': 4,
        'language': 'chinese',
        'speed': 1,
        'tone': 1,
        'volume': 1}

data5 = {'text': '关于西藏的传说有很多.',
        'model': 5,
        'language': 'chinese',
        'speed': 1,
        'tone': 1,
        'volume': 1}

data6 = {'text': '关于西藏的传说有很多.',
        'model': 6,
        'language': 'chinese',
        'speed': 1,
        'tone': 1,
        'volume': 1}
    
datas = [data1, data2, data3, data4, data5, data6]
#response = requests.post(test_url, json=json.dumps(data), headers=headers)
for data in datas:
    response = requests.post(test_url, json=data, headers=headers)
    with open('tmp/client/client' + str(data['model']) + '.wav', 'wb') as f:
        f.write(response.content)
