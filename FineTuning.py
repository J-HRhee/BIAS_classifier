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

# %%---------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------

import random
import pandas as pd
random.seed(40)
#AUG = []

def df_augmented(df, PUNC_RATIO=0.3,
					PUNCTUATIONS= ['.', ',', '!', '?', ';', ':']):
  AUG = []
  for i in range(len(df)):
    sent = str(df.sentence.values[i]).split()
    AUG.append(sent)

    q = random.randint(1, int(PUNC_RATIO * len(sent) + 1)) # 1부터 문장길이/3 까지 난수 생성
    qs = random.sample(range(0, len(sent)), q)             # 위에서 생성한 난수만큼 샘플링(문장부호 삽입위치 결정)
			
    for j in range(len(qs)):
      AUG[i].insert(qs[j], PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
    AUG[i] = " ".join(AUG[i]) # 다시 문자열로 변환
  
  AUG = pd.DataFrame(AUG, columns = ['sentence'])
  AUG['label'] = list(df['label'])
  return AUG

# %%
# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

import random
from random import shuffle
random.seed(1)

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
#import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet 

def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words

def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words)
	return new_words

def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=2):
	
	sentence = get_only_chars(sentence)
	words = sentence.split(' ')
	words = [word for word in words if word is not '']
	num_words = len(words)
	
	augmented_sentences = []
	num_new_per_technique = int(num_aug/4)+1

	#sr
	if (alpha_sr > 0):
		n_sr = max(1, int(alpha_sr*num_words))
		for _ in range(num_new_per_technique):
			a_words = synonym_replacement(words, n_sr)
			augmented_sentences.append(' '.join(a_words))

	#ri
	if (alpha_ri > 0):
		n_ri = max(1, int(alpha_ri*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_insertion(words, n_ri)
			augmented_sentences.append(' '.join(a_words))

	#rs
	if (alpha_rs > 0):
		n_rs = max(1, int(alpha_rs*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_swap(words, n_rs)
			augmented_sentences.append(' '.join(a_words))

	#rd
	if (p_rd > 0):
		for _ in range(num_new_per_technique):
			a_words = random_deletion(words, p_rd)
			augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	shuffle(augmented_sentences)

	#trim so that we have the desired number of augmented sentences
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#append the original sentence
	augmented_sentences.append(sentence)

	return augmented_sentences

# %%--#########################################################################################
EDA_aug_sent = []
EDA_aug_label = []

for i in range(len(train_valid)):
  EDA_aug_sent.append(eda(train_valid['sentence'].iloc[i]))
  EDA_aug_sent[i].pop()
  #EDA_aug_label.append(train_valid['label'].iloc[i])

EDA_aug_sent = pd.DataFrame(EDA_aug_sent, columns=['aug1','aug2'])
EDA_aug_sent = pd.concat([EDA_aug_sent['aug1'], EDA_aug_sent['aug2']], axis=0)
EDA_aug_sent = pd.DataFrame(EDA_aug_sent, columns=['sentence'])
EDA_aug_sent.reset_index(inplace=True, drop=True)

EDA_aug_label = pd.concat([train_valid['label'], train_valid['label']], axis=0)
EDA_aug_label = pd.DataFrame(EDA_aug_label, columns=['label'])
EDA_aug_label.reset_index(inplace=True, drop=True)

EDA_aug = pd.concat([EDA_aug_sent, EDA_aug_label], axis=1)

EDA_aug

# %%

ADEA_aug = df_augmented(train_valid)
#AUG
#AUG_1=df_augmented(train_valid)
#AUG_2=df_augmented(train_valid)

# %%---#########################################################################################

AUG = pd.concat([ADEA_aug, EDA_aug], axis=0)
df_aug = pd.concat([train_valid, AUG], axis=0)

# 1. df + aug + noise 10000
#df = pd.concat([df_aug, train_noise.iloc[:10000]], axis=0)

# 2. df + aug
df = df_aug

# 3. df 
#df = train_valid

df

# %% #################################################################################
# 
sentences=df.sentence.values
labels=df.label.values

# %% ################################################################################
# 문장에 cls, sep 토큰 추가(!중요: 최초 1회만 실행할 것!)---------------------------
# We need to add special tokens at the beginning and end of each sentence for BERT to work properly

sentences =  ["[CLS] " + sentence + " [SEP]" for sentence in sentences]

# %% ################################################################################
# 토크나이징 ---------------------------------------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print ("Tokenize the first sentence:")
print (tokenized_texts[0])

# %% ################################################################################
# Maximun Sequence length---------------------------------------------------
MAX_LEN = 128
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# Pad our input tokens
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# %% #####################################################################------######-
# Create attention masks
attention_masks= []

# Create a mask of 1s for each token followed by 0s for padding

for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

# %%-#####################################################################################
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                            random_state=2018, test_size=0.2)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.2)

# %% ########################################################################################
# Convert all of our data into torch tensors,-------------------
# the required datatype for our model

train_inputs = torch.tensor(train_inputs).to(torch.int64)
validation_inputs = torch.tensor(validation_inputs).to(torch.int64)
train_labels = torch.tensor(train_labels).to(torch.int64)
validation_labels = torch.tensor(validation_labels).to(torch.int64)
train_masks = torch.tensor(train_masks).to(torch.int64)
validation_masks = torch.tensor(validation_masks).to(torch.int64)

# %%-####################################################################################
batch_size=32

# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 
# with an iterator the entire dataset does not need to be loaded into memory

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# %% ##########################################################################################
# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.cuda()

# %% ####################################################################################
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# %% #########################################################################################
# 옵티마이저 설정-----------------------------------------------------------
# This variable contains all of the hyperparemeter information our training loop needs
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=1e-5,
                     warmup=.1)

# %%#########################################################################################-
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# %% ---#########################################################################################
# Store our loss and accuracy for plotting
train_loss_set = []

# Number of training epochs (authors recommend between 2 and 4)
epochs = 2

# trange is a tqdm wrapper around the normal python range
for _ in trange(epochs, desc="Epoch"):
  
  
  # Training
  
  # Set our model to training mode (as opposed to evaluation mode)
  model.train()
  
  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0
  
  # Train the data for one epoch
  for step, batch in enumerate(train_dataloader):
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Clear out the gradients (by default they accumulate)
    optimizer.zero_grad()
    # Forward pass
    loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    train_loss_set.append(loss.item())    
    # Backward pass
    loss.backward()
    # Update parameters and take a step using the computed gradient
    optimizer.step()
    
    
    # Update tracking variables
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1

  print("Train loss: {}".format(tr_loss/nb_tr_steps))
    
    
  # Validation

  # Put model in evaluation mode to evaluate loss on the validation set
  model.eval()

  # Tracking variables 
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  # Evaluate data for one epoch
  for batch in validation_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      eval_loss += loss.item()
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

  #print("Train loss: {}".format(tr_loss/nb_tr_steps))
  print("Valid loss: {}".format(eval_loss / nb_eval_steps))
  print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

# %%---------------------------------------------------------------------------------------
plt.figure(figsize=(15,8))

plt.title("Training loss")
plt.xlabel("Batch")
plt.ylabel("Loss")
plt.plot(train_loss_set)#, valid_loss_set)
plt.show()

# %%---------------------------------------------------------------------------------------

# %%-------------------------------------------------------------------------------------------------

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

# %%-- Prediction on test set-------------------------------------------------------

# Put model in evaluation mode
model.eval()

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
    logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

# %% ------------------------------------------------------
  # Import and evaluate each test batch using Matthew's correlation coefficient
from sklearn.metrics import matthews_corrcoef
matthews_set = []

for i in range(len(true_labels)):
    matthews = matthews_corrcoef(true_labels[i],
                 np.argmax(predictions[i], axis=1).flatten())
    matthews_set.append(matthews)

# %% -----------------------------------------------------------------------------
# Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
flat_predictions = [item for sublist in predictions for item in sublist]
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
flat_true_labels = [item for sublist in true_labels for item in sublist]

#matthews_corrcoef(flat_true_labels, flat_predictions)

# 1에 가까울수록 완벽한 예측
# 0에 가까울수록 임의 예측
# -1에 가까울수록 역 예측

# %%-----------------------------------------------
from sklearn.metrics import classification_report
print(classification_report(flat_true_labels, flat_predictions))



# %%----할 일
# 1. 문장을 예측한 fn_pred 생성
# 2. fn_pred의 기사인덱스마다 positive rate 구하기
# 3. positive rate와 fakenews label과 비교하기

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

model.eval()

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
    logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

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
# 검증결과: statistic = 3.72
#          P value = 0.0002 < 0.05            

# %%

torch.save(model, 'D:/WIKIBIAS/model_AEDA_EDA.pt')



# %%
#model_c = load_model('model_AEDA_noise')

#model = TheModelClass(*args, **kwargs)
#model.load_state_dict(torch.load(PATH))

model_c = torch.load('model_AEDA_noise.pt',map_location=device)
model_c.eval()

# %%
# Put model in evaluation mode
model_c.eval()

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
# %%











