import string
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


#source_path = "../../TeamDatasets/az-en/az-en-tanzil/Tanzil.az-en.az"
#target_path = "../../TeamDatasets/az-en/az-en-tanzil/Tanzil.az-en.en"

source_path = "../../TeamDatasets/az-tr/Tanzil.az-tr.az"
target_path = "../../TeamDatasets/az-tr/Tanzil.az-tr.tr"

#source_path = "../../TeamDatasets/en-tr/Tanzil.en-tr.tr"
#target_path = "../../TeamDatasets/en-tr/Tanzil.en-tr.en"
#


source_txt = open(source_path, encoding='utf8').read().split('\n')

target_txt = open(target_path, encoding='utf8').read().split('\n')


target_txt = target_txt[:120000]
source_txt = source_txt[:120000]

def text_cleaning(data): 
    data = [s.translate(str.maketrans('','',string.punctuation)) for s in data]
    data = [' '.join(s.split()) for s in data]
    return data


target_txt = text_cleaning(target_txt)
source_txt = text_cleaning(source_txt)


    

raw_data = {'Source_lang': [sent for sent in source_txt],
            'Target_lang': [sent for sent in target_txt]}

df = pd.DataFrame(raw_data, columns=['Source_lang','Target_lang'])


#df.to_csv('df_test.csv',index = False)
#train, test = train_test_split(df, test_size = 0.2)

#train.to_csv('tr_en_train20k.csv',index = False)
#test.to_csv('tr_en_test20k.csv',index = False)

train = df.iloc[:100000,:]
test = df.iloc[100000:,:]

srcData = test['Source_lang']
trgData = test['Target_lang']

with open("Datasets/az_tr_azer_100k.txt", 'w',encoding='utf-8') as op:
    for each in srcData:
        op.write(each + '\n')

with open("Datasets/az_tr_turk_100k.txt",'w',encoding = 'utf-8') as op:
    for each in trgData:
        op.write(each + '\n')

train.to_csv('Datasets/az_tr_train100k.csv',index = False)
test.to_csv('Datasets/az_tr_test100k.csv',index = False)