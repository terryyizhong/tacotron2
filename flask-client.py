import requests
import numpy as np
import json

addr = 'http://localhost:5000'
test_url = addr + '/api/test'
content_type = 'application/json'
headers = {'content-type': content_type}

data = {'data': 'hello.'}
data2 = {'data': 'how are you doing'}

response = requests.post(test_url, json=json.dumps(data), headers=headers)
with open('client.wav', 'wb') as f:
    f.write(response.content)

#response = requests.post(test_url, json=json.dumps(data2), headers=headers)
