import numpy as np
import soundfile as sf
from glob import glob
from py_activlev import asl_P56
from tqdm import tqdm
import json
data_path = "/home/bjwoo/PycharmProjects/data/sitec"

# with open("mix.json", "r") as st_json:
#     st_python = json.load(st_json)
# asdf = dict(st_python)
# print("")
data_list = glob(data_path+"/**/*.wav",recursive=True)
result_file = []

for data in tqdm(data_list):
    y, fs = sf.read(data)
    asl_msq, actfact, c0 = asl_P56(y,fs,16)
    name = data.split("/")[-2]+"/"+data.split("/")[-1]
    result_file.append((name , asl_msq))

with open("mix.json", "w") as f:
    json.dump(result_file, f, indent=4)


