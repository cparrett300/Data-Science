
import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json=dict(zip(['Section-1','Section-2','Section-3','Section-4','Section-5','Section-6','Section-7','Section-8','Section-9'],
[1,1589.70,1400.60,9046.19,47.47,8.4195,392,39.06,23.4190])))

print(r.json())