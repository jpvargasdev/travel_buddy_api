import pandas as pd
import requests
from io import StringIO

url="https://docs.google.com/spreadsheets/d/1lkYbOQSXlvOd3bZu2v9_wgFrzUI6U5v0PNmV_3z9Okg/edit?usp=sharing"

file_id = url.split('/')[-2]
dwn_url='https://drive.google.com/uc?export=download&id=' + file_id
url2 = requests.get(dwn_url).text
csv_raw = StringIO(url2)
df = pd.read_csv(csv_raw)
print(df.head())


