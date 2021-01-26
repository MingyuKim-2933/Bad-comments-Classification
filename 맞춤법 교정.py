import csv
import sys
import pandas as pd
import numpy as np
import torch
from konlpy.tag import Komoran
from hanspell import spell_checker

dataset = []

csv.field_size_limit(100000000)
with open('./악플_test.csv', 'r', encoding = 'utf-8') as File:
    csv_reader = csv.reader(File, delimiter = ',')
    for row in csv_reader:
        dataset.append(row)
# print(dataset[0])
# print(dataset[1])
# print(dataset[0][0])
# print(dataset[0][1])
length = len(dataset)
print(length)
for i in range(length):
    
    dataset[i][0]=dataset[i][0].replace("\n", '\\n').replace("\t", "\\t").replace("&", "&amp;")
    print(i, dataset[i][0])
    result = spell_checker.check(dataset[i][0])
    
    dataset[i][0] = result.checked
    print("결과 :"+dataset[i][0])
    print(i)

# dataset[20071][0]=dataset[20071][0].replace("\n", '\\n').replace("\t", "\\t").replace("&", "&amp;")
# print("20071"+ dataset[20071][0])
# result = spell_checker.check(dataset[20071][0])
# dataset[20071][0] = result.checked
# print("결과 :"+dataset[20071][0])

with open('./악플맞춤법_test.csv', 'wt', encoding = 'utf-8-sig') as f:
    writer = csv.writer(f, delimiter = ',', lineterminator = '\n')
    writer.writerows(dataset)
