import requests
import time
headers={"Cookie": 'COOKIE HERE'}
idx = IDX
url = "https://cs109b-nb.seas.harvard.edu/user/"+str(idx)+"/tree?"
serial = 1523644395397
while True:
    requests.get(url,headers=headers,verify=False)
    lc = "https://cs109b-nb.seas.harvard.edu/user/"+str(idx)+"/api/contents?type=directory&_="+str(serial)
    print(requests.get(lc,headers=headers,verify=False).text)
    serial += 3
    time.sleep(5)
