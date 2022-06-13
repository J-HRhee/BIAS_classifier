# %% 라이브러리 불러오기----------------------------------------------
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

#torch.cuda.is_available()
# %% GPU 확인-----------------------------------------------------------
device = torch.device("cuda")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

# %% 데이터셋 로딩(train, dev, test가 이미 있어서 따로)----------------
train_noise = pd.read_csv("D:/WIKIBIAS/data/class_binary/noisy_train.tsv", delimiter='\t', names=["sentence","nan", "label"])
train = pd.read_csv("D:/WIKIBIAS/data/class_binary/train.tsv", delimiter='\t', names=["sentence","nan", "label"])
dev= pd.read_csv("D:/WIKIBIAS/data/class_binary/dev.tsv", delimiter='\t', names=["sentence","nan", "label"])
test = pd.read_csv("D:/WIKIBIAS/data/class_binary/test.tsv", delimiter='\t', names=["sentence","nan", "label"])

train_noise.drop(['nan'], axis=1, inplace=True)
test.drop(['nan'], axis=1, inplace=True)


train_valid = pd.concat([dev, train], axis=0)
train_valid.drop(['nan'], axis=1, inplace=True)

# %% #################################################

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# 테스트셋 불러오기

df= test#.drop(['nan'], axis=1, inplace=True)
# Create sentence and label lists
sentences = test.sentence.values

# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
labels = df.label.values

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]


MAX_LEN = 128

# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask) 

prediction_inputs = torch.tensor(input_ids).to(torch.int64)
prediction_masks = torch.tensor(attention_masks).to(torch.int64)
prediction_labels = torch.tensor(labels).to(torch.int64)
  
batch_size = 32  


prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
# %%#################################################################
# 학습된 모델 로드

model_c = torch.load('model_AEDA_noise.pt',map_location=device)
model_c.eval()
# %% ##################################################################################
# 테스트셋으로 성능체크


# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  # Telling the model not to compute or store gradients, saving memory and speeding up prediction
  with torch.no_grad():
    # Forward pass, calculate logit predictions
    logits = model_c(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

# %% -----------------------------------------------------------------------------
# Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

from sklearn.metrics import classification_report
print(classification_report(flat_true_labels, flat_predictions))
# %%
# %% --------------------------------------------------------------------------------

import pandas as pd
import numpy as np

politifact = pd.read_csv("D:/WIKIBIAS/data/fake_news_dataset/politifact.csv")
politifact_post = politifact[(politifact['sources'].str.contains('image')==False) &
                             (politifact['sources'].str.contains('video')==False) &
                             (politifact['sources_post_location'].str.contains('video')==False)]# &
                             #(politifact['fact'].str.contains('flip')==False) &
                             #(politifact['fact'].str.contains('flop')==False)] 
                             
politifact_text = politifact_post['sources_quote'].str.replace(pat = '\n', repl=r'', regex=True)
politifact_label = politifact_post['fact']


# %%################################################################################
# ---------------------------------------------------------
from nltk.tokenize import sent_tokenize
import re

politifact_tokenized = []
for i in range(len(politifact_text)):
  politifact_tokenized.append(sent_tokenize(re.sub('\n',"",str(politifact_text.iloc[i]))))


# %%--######################################-----역 정규화 -------------------------
#

politifact_tokenized_sentence = []


for i in range(len(politifact_tokenized)):
  n = len(politifact_tokenized[i])
  for j in range(n):
    politifact_tokenized_sentence.append(politifact_post['Unnamed: 0'].iloc[i]) # 문장의 기사 id
    politifact_tokenized_sentence.append(politifact_tokenized[i][j]) #

# %%-###############################################################################
# ##################################################################################

politifact_sentence_df = pd.DataFrame(np.array(politifact_tokenized_sentence).reshape(-1,2),columns=['news_id','sentence'])
politifact_sentence_df # 문서단위

# %% #################################################################################
# ####################################################################################

# 페이크뉴스 실험
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
sentences = politifact_sentence_df.sentence.values

# We need to add special tokens at the beginning and end of each sentence for BERT to work properly
sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
#labels = fn_sentence_df.label.values

tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]


MAX_LEN = 128

# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Create attention masks
attention_masks = []

# Create a mask of 1s for each token followed by 0s for padding
for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask) 

prediction_inputs = torch.tensor(input_ids).to(torch.int64)
prediction_masks = torch.tensor(attention_masks).to(torch.int64)
#prediction_labels = torch.tensor(labels).to(torch.int64)
  
batch_size = 32  


prediction_data = TensorDataset(prediction_inputs, prediction_masks)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


# %%----#############################################################
# ------------------------------------------------------------------------

model_c.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask = batch
  # Telling the model not to compute or store gradients, saving memory and speeding up prediction
  with torch.no_grad():
    # Forward pass, calculate logit predictions
    logits = model_c(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  #label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  # true_labels.append(label_ids)

# %%------------------------------------------------------------------------------

flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
#flat_true_labels = [item for sublist in true_labels for item in sublist]
#print(classification_report(flat_true_labels, flat_predictions))
len(flat_predictions)

# %%
politifact_prediction = pd.DataFrame(flat_predictions, columns=['prediction'])
fake_news_labeled = pd.concat([politifact_sentence_df, politifact_prediction], axis=1)
fake_news_labeled['news_id'] = pd.to_numeric(fake_news_labeled['news_id'])

fakenews_biased = fake_news_labeled.groupby('news_id').mean()
fakenews_biased



# %%
#pd.concat([fn_train,fake_news_labeled], axis=1)
DF = pd.merge(politifact_post, fakenews_biased, how='inner', left_on = 'Unnamed: 0', right_on = 'news_id')
DF

# %%
import scipy.stats


M = DF[DF['fact']=='pants-fire']['prediction']
N = DF[DF['fact']=='true']['prediction']
scipy.stats.ttest_ind(M, N, equal_var=False)
# %%
