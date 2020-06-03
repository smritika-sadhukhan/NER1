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
df = pd.read_csv('/content/NER1/Ner_pack/Dataset/df1_train', encoding = "ISO-8859-1")

word_to_index=pickle.load(open('/content/NER1/Ner_pack/ner_files/word_to_index_2.pickle','rb'))
tag_to_index=pickle.load(open('/content/NER1/Ner_pack/ner_files/tag_to_index_2.pickle','rb'))
model= load_model('/content/NER1/Ner_pack/model/model_2.h5',custom_objects={'CRF':CRF, 
                                                  'crf_loss':crf_loss, 
                                                  'crf_viterbi_accuracy':crf_viterbi_accuracy})
idx2word = {i: w for w, i in word_to_index.items()}
idx2tag = {i: w for w, i in tag_to_index.items()}
words = list(df['words'].unique())
tags = list(df['tags'].unique())



def AI_entity(s):
  dict1={}
  if type(s)==str:
     print('The entities are ...')
  else:
    print('Sorrry !!!! Datatype did not match')
    print('Please give a string input')
    print(' '*80)
    print('    For example')
    print('*'*80)
    print('"I want to book the show at 12 am on 25th july"')
    print('*'*80)

    return
  
  list8=[]
  list10=[]
  for i in s.split():
    list8.append(word_to_index.get(i,1))
    list10.append(i)

  X = pad_sequences(maxlen = 15, sequences = [list8], padding = "post", value = word_to_index["PAD"])

  p = model.predict(np.array(X))
  p = np.argmax(p, axis=-1)
  c=-1
  for w, pred in zip(X[0], p[0]):
    c=c+1
    if w != 0 and idx2tag[pred]!='PAD'  and idx2tag[pred]!='O':
         try:
           dict1[list10[c]]=idx2tag[pred]

         except:
            break

  return(dict1)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

