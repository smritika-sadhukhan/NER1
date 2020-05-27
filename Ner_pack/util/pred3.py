# -*- coding: utf-8 -*-
"""pred2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xc5Kf9SHYqA11i0lxpR5Uvp4xbkG4Jx5

# **IMPORT**
"""


# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras_contrib.layers import CRF
import pickle
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from keras.utils import to_categorical
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras.models import Model, Input
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import TweetTokenizer
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import f1_score
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from keras.preprocessing.text import text_to_word_sequence
import pickle
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras_contrib.layers import CRF
import pickle
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

from keras.utils import to_categorical
from keras.layers import LSTM, Dense, TimeDistributed, Embedding, Bidirectional
from keras.models import Model, Input
from keras.callbacks import ModelCheckpoint
from nltk.tokenize import TweetTokenizer
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import f1_score
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from keras.preprocessing.text import text_to_word_sequence
import pickle



"""# **LOAD DATA**"""
diri='/content/Ner_pack/'
df = pd.read_csv(diri+'ner_dataset.csv', encoding = "ISO-8859-1")
word_to_index=pickle.load(open(diri+'word_to_index.pickle','rb'))
tag_to_index=pickle.load(open(diri+'tag_to_index.pickle','rb'))
model= load_model(diri+"model1.h5",custom_objects={'CRF':CRF, 
                                                  'crf_loss':crf_loss, 
                                                  'crf_viterbi_accuracy':crf_viterbi_accuracy})

idx2word = {i: w for w, i in word_to_index.items()}
idx2tag = {i: w for w, i in tag_to_index.items()}
words = list(df['Word'].unique())
tags = list(df['Tag'].unique())

sw=['i','me','them','they','himself','herslf','yourself','myself','their','here','of','on']
loc="india west bengal kolkata orrisa gujrath hyderabad mumbai maharashtra goa andra pradesh indore delhi noida himachal pradesh arunachal pradesh kerela vellore bangalore shimla karnataka".lower()
date='january february march april may june july august september october november december'
money='rs rupees $ dollars euro pound rupiah bucks'
day='1st 2nd 3rd 4th 5th 6th 7th 8th 9th 10th 11th 12th 13th 14th 15th 16th 17th 18th 19th 20th 21st 22nd 23rd 24th 25th 26th 27th 28th 29th 30th 31st  sunday monday tuesday wednesday thursday friday saturday '

"""# **PREDICTION**"""

# Commented out IPython magic to ensure Python compatibility.
# %time AI_entity('i want to fly from mumbai to kolkata on 2nd may with rs 3 2pm')

def power(x,y):
    return x**y

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


from nltk.tokenize import TweetTokenizer
dict={}

def AI_RB_entity(s):
  ss=s.lower()
  tknzr = TweetTokenizer()
  s3=tknzr.tokenize(ss)
  s1=[]
  s2=[]
  for i in s3:
    if i not in sw:
       s2.append(i)
  for i in s2:
      if word_to_index.get(i)==None:
          i='UNK'
      s1.append(i)
 
  import re
  mem1=0
  mem2=0
  dict={}
  for i in range(0,len(s1)):
    if s1[i]=='from':
        mem1=mem1+1
    elif s1[i]=='to':
        mem2=mem2+1
    elif s1[i]=='UNK':
        if mem1>0:
            if len(re.findall(s2[i],loc))>0:
                mem1=mem1-1
                dict['start']=s2[i]
            else:
                dict['person']=s2[i]
    
        elif mem2>0:
            if len(re.findall(s2[i],loc))>0:
                mem2=mem2-1

                dict['destination']=s2[i]
                mem2=0
        
  
    elif len(re.findall(s2[i],day))>0:
        if i+1<len(s2):
          if  len(re.findall(s2[i+1],date))>0:
              dict['date']=' '.join([s2[i],s2[i+1]])
          elif s2[i+1] =='am' or s2[i+1]=='pm':
              dict['time']=' '.join([s2[i],s2[i+1]])

    if s2[i].isdigit():
        if len(re.findall(s2[i-1],money))>0:
           dict['money']=' '.join([s2[i],s2[i-1]])

        elif  i+1<len(s2):
           if len(re.findall(s2[i+1],money))>0:
              dict['money']=' '.join([s2[i],s2[i+1]])

  
  X = [word_to_index.get(i,1) for i in s.split()]
  X = pad_sequences(maxlen = 75, sequences = [X], padding = "post", value = word_to_index["PAD"])
  p = model.predict(np.array(X))
  p = np.argmax(p, axis=-1)
  print("{:15}||{}".format("Word", "Prediction"))
  print(30 * "=")
  for w, pred in zip(X[0], p[0]):
     if w != 0 and  idx2tag[pred]!='PAD' and  idx2tag[pred]!='O' and words[w-2]!='UNK':
        print("{:15}: {}".format(words[w-2], idx2tag[pred]))
        dict[idx2tag[pred]]=words[w-2]

  
  

  return(dict)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


from nltk.tokenize import TweetTokenizer
dict={}

def RB_entity(s):
  s=s.lower()
  tknzr = TweetTokenizer()
  s3=tknzr.tokenize(s)
  
  # nlp = en_core_web_sm.load()
  # doc=nlp(s)
  # list1=[]
  # for X in doc:
  #   list1.append((str(X).lower()))
  # s3=list1

  s1=[]
  s2=[]
  for i in s3:
    if i not in sw:
       s2.append(i)
  for i in s2:
      if word_to_index.get(i)==None:
          i='unk'
      s1.append(i)
 
  import re
  mem1=0
  mem2=0
  dict={}
  for i in range(0,len(s1)):
    if s1[i]=='from':
        mem1=mem1+1
    elif s1[i]=='to':
        mem2=mem2+1
    elif s1[i]=='unk':
        if mem1>0:
            if len(re.findall(s2[i],loc))>0:
                mem1=mem1-1
                dict['start']=s2[i]
            else:
                dict['person']=s2[i]
    
        elif mem2>0:
            if len(re.findall(s2[i],loc))>0:
                mem2=mem2-1

                dict['destination']=s2[i]
                mem2=0
        
  
    elif len(re.findall(s2[i],day))>0:
        if i+1<len(s2):
          if  len(re.findall(s2[i+1],date))>0:
              dict['date']=' '.join([s2[i],s2[i+1]])
          elif s2[i+1] =='am' or s2[i+1]=='pm':
              dict['time']=' '.join([s2[i],s2[i+1]])

    if s2[i].isdigit():
        if len(re.findall(s2[i-1],money))>0:
           dict['money']=' '.join([s2[i],s2[i-1]])

        elif  i+1<len(s2):
           if len(re.findall(s2[i+1],money))>0:
              dict['money']=' '.join([s2[i],s2[i+1]])



  

  return(dict)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

from nltk.tokenize import TweetTokenizer
dict={}

def AI_entity(s):
 
  X = [word_to_index.get(i,1) for i in s.split()]
  X = pad_sequences(maxlen = 75, sequences = [X], padding = "post", value = word_to_index["PAD"])
  p = model.predict(np.array(X))
  p = np.argmax(p, axis=-1)
  print("{:15}||{}".format("Word", "Prediction"))
  print(30 * "=")
  for w, pred in zip(X[0], p[0]):
     if w != 0 and  idx2tag[pred]!='PAD' and  idx2tag[pred]!='O' and words[w-2]!='UNK':
        print("{:15}: {}".format(words[w-2], idx2tag[pred]))
        dict[idx2tag[pred]]=words[w-2]

  

  return(dict)