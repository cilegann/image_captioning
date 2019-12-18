#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import numpy as np
from keras.models import Model,load_model
from keras.layers import Input, LSTM, Dense,Activation,Embedding,merge,TimeDistributed
from keras.layers.core import Lambda
from keras.layers import concatenate,Reshape,Concatenate,RepeatVector,add
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import plot_model
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from nltk.translate.bleu_score import sentence_bleu
from keras.callbacks import EarlyStopping,ModelCheckpoint
import random
from PIL import Image
from matplotlib import pyplot as plt
import pickle
from keras import backend as K
import math
import random


# In[2]:


coco_json='./data/dataset_coco.json'
tokenfile='./token_tune.pkl'
detokenfile='./detoken_tune.pkl'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.Session(config=config)
KTF.set_session(session)

batch_size=64
seq_len=10 #49+'\t'+'\n'
# In[3]:


with open(coco_json) as file:
    lines=file.readlines()
data=json.loads(lines[0])['images']


# In[4]:


train_x_file=[]
train_y_seq=[]
vali_x_file=[]
vali_y_seq=[]
token={}
# if os.path.exists(tokenfile):
#     with open(tokenfile,'rb') as file:
#         token=pickle.load(file)
# else:
token['\t']=1
token['\n']=2


# In[5]:

print("Building token...")
for d in data:
    if d['split']=='train':
        for sen in random.sample(d['sentences'],k=2):
            seq=sen['tokens']
            if len(seq)>seq_len:
                continue
            train_x_file.append('./coco/'+d['filepath']+'/'+d['filename'])
            for s in seq:
                if s.lower() not in token:
                    token[s.lower()]=len(token)+1
            train_y_seq.append(['\t']+[s.lower() for s in seq[:seq_len]]+['\n'])
    if d['split']=='test':
        vali_x_file.append('./coco/'+d['filepath']+'/'+d['filename'])
        # for sen in d['sentences']:
        #     vali_x_file.append('./coco/'+d['filepath']+'/'+d['filename'])
        #     seq=sen['tokens']
        #     for s in seq:
        #         if s not in token:
        #             token[s]=len(token)+1
        #     vali_y_seq.append(['\t']+seq+['\n'])


# In[6]:


with open(tokenfile,'wb') as file:
    pickle.dump(token,file)

# if os.path.exists(detokenfile):
#     with open(detokenfile,'rb') as file:
#         detoken=pickle.load(file)
# else:
detoken=dict((i,char) for char,i in token.items())
with open(detokenfile,'wb') as file:
    pickle.dump(detoken,file)


# In[7]:


num_tokens=len(token)

c = list(zip(train_x_file, train_y_seq))
random.shuffle(c)
train_x_file, train_y_seq = zip(*c)
train_x_file=train_x_file[:15000]
train_y_seq=train_y_seq[:15000]
train_x_img=[]
train_x_seq=[]
train_y=[]
print("Building data...")
for i in range(len(train_x_file)):
    for t in range(len(train_y_seq[i])):
        if t>0:
            did=np.zeros(
                (seq_len+2),dtype='float32'
            )
            dtd=np.zeros(
                (1),dtype='float32'
            )
            for tt in range(t):
                did[tt]=float(token[train_y_seq[i][tt]])
            dtd[0]=float(token[train_y_seq[i][t]])
            train_x_img.append(train_x_file[i])
            train_x_seq.append(did)
            train_y.append(dtd)
#c=list(zip(train_x_img,train_x_seq,train_y))
#random.shuffle(c)
#train_x_img,train_x_seq,train_y=zip(*c)
print("Num of train:",len(train_x_img))
print("Num of vali:",len(vali_x_file))
print("Num of token:",num_tokens)


# In[ ]:


vgg = VGG16(weights='imagenet')
vgg.layers.pop()
vggmodel=Model(inputs=vgg.inputs,outputs=vgg.layers[-1].output)


# In[10]:


dgenverbose=0
def dataGenerator(x,xx,y):
    while(1):
        decoder_input_data=[]
        decoder_target_data=[]
        encoder_input_data=[]
        for i in range(len(x)):
            img=image.load_img(x[i], target_size=(224, 224))
            img = image.img_to_array(img)
#             img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            encoder_input_data.append(img)
            decoder_input_data.append(xx[i])
            decoder_target_data.append(y[i])
            if len(decoder_input_data)==batch_size:
                yield ([np.array(encoder_input_data),np.array(decoder_input_data)],np.array(decoder_target_data))
                decoder_input_data=[]
                decoder_target_data=[]
                encoder_input_data=[]


# In[11]:

print("Building model...")
latent_dim=128

img_input=Input (shape=(224,224,3,) )
vgg = VGG16(weights='imagenet')
vgg.layers.pop()
vggmodel=Model(inputs=vgg.inputs,outputs=vgg.layers[-1].output)


encode_dense=Dense( latent_dim, activation='relu' )( vggmodel(img_input) )
encode_dense=RepeatVector( seq_len+2 )( encode_dense )

decode_input=Input( shape=(seq_len+2,) )
decode_emb=Embedding( num_tokens+1, 128, mask_zero=True, input_length=(seq_len+2) )( decode_input )
decoder_LSTM=LSTM (latent_dim*2, return_sequences=True )( decode_emb )
decoder_timedistri=TimeDistributed( Dense( latent_dim ) )( decoder_LSTM )
decoder_vec=concatenate( [encode_dense, decoder_timedistri] )
decoder=LSTM( latent_dim*2 )( decoder_vec )
decoder=Dense( num_tokens, activation='softmax' )( decoder )
model=Model( [img_input, decode_input], decoder )
model.summary()

# In[11]:


def ppx(y_true, y_pred):
     loss = K.sparse_categorical_crossentropy(y_true, y_pred)
     perplexity = K.cast(K.pow(math.e, K.mean(loss, axis=-1)), K.floatx())
     return perplexity


# In[12]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])


# In[18]:

cbckpt = ModelCheckpoint('vgg_tune_best_weight.h5',monitor='loss', verbose=1, save_best_only=True, mode='min')
cbes=EarlyStopping(patience=2,monitor='loss')
try:

    model.fit_generator(
        dataGenerator(train_x_img,train_x_seq,train_y),
        #validation_data=dataGenerator(vali_x_file,vali_y_seq),
        steps_per_epoch=((len(train_x_img))//batch_size),
        #steps_per_epoch=10000,
        #validation_steps=1,
        epochs=200,
        callbacks=[cbckpt]
    )
except KeyboardInterrupt:
    model.save_weights('./vgg_tune_keyboardInterrupt_saved_weight.h5')


# In[17]:


print("Train done")
model.save_weights('./vgg_tune_model_weight.h5')
# plot_model(model,to_file='./model_struct2.jpg')


