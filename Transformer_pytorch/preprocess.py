import string
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

#
#source_path = "../../TeamDatasets/az-en/az-en-tanzil/Tanzil.az-en.az"
#target_path = "../../TeamDatasets/az-en/az-en-tanzil/Tanzil.az-en.en"

#source_path = "../../TeamDatasets/az-tr/az-tr-tanzil/Tanzil.az-tr.az"
#target_path = "../../TeamDatasets/az-tr/az-tr-tanzil/Tanzil.az-tr.tr"

source_path = "../../TeamDatasets/tr-en/tr-en-tanzil/Tanzil.en-tr.tr"
target_path = "../../TeamDatasets/tr-en/tr-en-tanzil/Tanzil.en-tr.en"

#backTranslation
#target_path = "../../TeamDatasets/tr-en/tr-en-tanzil/Tanzil.en-tr.tr"
#source_path = "../../TeamDatasets/tr-en/tr-en-tanzil/Tanzil.en-tr.en"

##Tatoeba
#source_path = "../../TeamDatasets/az-en/az-en-tatoeba/Tatoeba.az-en.az"
#target_path = "../../TeamDatasets/az-en/az-en-tatoeba/Tatoeba.az-en.en"


#source_path = "../../TeamDatasets/az-en/az-en-tatoeba/Tatoeba.az-en.az"
#target_path = "../../TeamDatasets/az-en/az-en-tatoeba/Tatoeba.az-en.en"

#source_path = "../../TeamDatasets/az-en/az-en-tatoeba/Tatoeba.az-en.az"
#target_path = "../../TeamDatasets/az-en/az-en-tatoeba/Tatoeba.az-en.en"

source_txt = open(source_path, encoding='utf8').read().split('\n')

target_txt = open(target_path, encoding='utf8').read().split('\n')


#target_txt1 = target_txt[100000:110000]
#source_txt1 = source_txt[100000:110000]
#
#
#target_txt2 = target_txt[200000:210000]
#source_txt2 = source_txt[200000:210000]
#
#source_txt = source_txt1 + source_txt2
#target_txt = target_txt1 + target_txt2

target_txt = target_txt[:410000]
source_txt = source_txt[:410000]

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

train = df.iloc[:400000,:]
test = df.iloc[400000:,:]
#test = df

srcData = test['Source_lang'].tolist()
trgData = test['Target_lang'].tolist()

#with open("Testset/Tanzil/azer_20k_split.txt", 'w',encoding='utf-8') as op:
#    for each in srcData[:-1]:
#        op.write(each + '\n')
#    op.write(srcData[-1])
##
#with open("Testset/Tanzil/eng_20k_split.txt",'w',encoding = 'utf-8') as op:
#    for each in trgData[:-1]:
#        op.write(each + '\n')
#    op.write(trgData[-1])


train.to_csv('Datasets/Tanzil/tr_en_train400k.csv',index = False)
test.to_csv('Datasets/Tanzil/tr_en_test400k.csv',index = False)