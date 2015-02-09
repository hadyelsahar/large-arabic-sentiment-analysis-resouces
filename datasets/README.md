# Datasets description

this readme file contains naming convension of the dataset files 

## Datasets description and sizes :

- ATT.csv
  - Dataset of Attraction Reviews scrapped from TripAdvisor.com 
  - 2154 reviews

- HTL.csv
  - Dataset of Hotel Reviews scrapped from TripAdvisor.com 
  - 15572 reviews

- MOV.csv
  - Dataset of Movie Reviews scrapped from elcinema.com
  - 1524 reviews

- PROD.csv
  - Dataset of product reviews scrapped from souq.com
  - 4272 reviews


- RES1.csv
  - dataset of restaurant reviews scrapped from qaym.com
  - 8364 reviews

- RES2.csv
  - dataset of restaurant reviews scrapped from tripadvisor.com
  - 2642 reviews  
  
- RES.csv
  - RES1.csv and RES2.csv combined
  - 10970 reviews



## loading datasets 
using python & pandas you can easily load any of the datasets as following :
```
>> import pandas as pd 
>> x = pd.read_csv("HTL.csv",encoding="utf-8")
>> x.shape

(15572, 2)
```




