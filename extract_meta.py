

import os
import json
import csv
import pandas as pd

PATH_DATA="/Users/ezratampubolon/PythonProjects/Financial/edgar-data/companyfacts"
PATH_SAVE="meta_data.csv"

li_comp_file=os.listdir(PATH_DATA)

meta_data=[]

for file_nm in li_comp_file:
    with open(os.path.join(PATH_DATA,file_nm), encoding="utf-8") as file:
        # Define service
        raw_data=json.load(file)
        

        meta_data.append((file_nm,raw_data.get("cik"),raw_data.get("entityName")))

df=pd.DataFrame(meta_data,columns=["file_nm","cik","entity_name"])

df.to_csv(PATH_SAVE,index=False)