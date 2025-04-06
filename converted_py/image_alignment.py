#!/usr/bin/env python
# coding: utf-8


# importing important libraries
import pandas as pd
import os


# File containing the saved emoji data
PROJECT_ROOT = os.path.dirname(os.getcwd())
data_folder = os.path.join(PROJECT_ROOT, "data")
stickers_data = os.path.join(data_folder,"../data/flaticon_stickers.csv")



stickers = pd.read_csv(stickers_data)
stickers



from hashlib import md5
stickers['filename'] = stickers['image'].apply(lambda x: md5(x.encode()).hexdigest() + os.path.splitext(x)[-1])
stickers



stickers.to_csv()





