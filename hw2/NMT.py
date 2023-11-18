import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.data import Field, BucketIterator
import random
import numpy as np
from helper import TranslationDataset #read_file, tokenize, build_vocab
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from collections import defaultdict
from collections import Counter
from lstm import Encoder, Decoder

from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from underthesea import word_tokenize as word_tokenize_vi
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

def tokenize(text, lang='en'):
    if lang == 'en':
        return [tok for tok in word_tokenize(text.lower())]
    else:
        return [tok for tok in word_tokenize_vi(text.lower())]
        # return [tok for tok in word_tokenize(text.lower(), language='vi')]

def build_vocab(file_path):
    vocab = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for idx, word in enumerate(file):
            vocab[word.strip()] = idx
    return vocab

train_en = read_file('hw2_dataset/train.en')
train_vi = read_file('hw2_dataset/train.vi')
vocab_en = build_vocab('hw2_dataset/vocab.en')
vocab_vi = build_vocab('hw2_dataset/vocab.vi')
assert len(train_en) == len(train_vi), "Datasets do not have the same number of sentences"

for special_token in ['<pad>', '<unk>', '<sos>', '<eos>']:
    if special_token not in vocab_en:
        vocab_en[special_token] = len(vocab_en)
    if special_token not in vocab_vi:
        vocab_vi[special_token] = len(vocab_vi)

counter_en = Counter({word: 1 for word in vocab_en})  # Using 1 as a default frequency
counter_vi = Counter({word: 1 for word in vocab_vi})

vocab_en_obj = Vocab(counter_en, specials=['<unk>', '<pad>', '<sos>', '<eos>'])
vocab_vi_obj = Vocab(counter_vi, specials=['<unk>', '<pad>', '<sos>', '<eos>'])

SOURCE = Field(tokenize=lambda x: tokenize(x, lang='en'), init_token='<sos>', eos_token='<eos>', lower=True)
TARGET = Field(tokenize=lambda x: tokenize(x, lang='vi'), init_token='<sos>', eos_token='<eos>', lower=True)

SOURCE.vocab = vocab_en_obj
TARGET.vocab = vocab_vi_obj

# def collate_batch(batch):
#     pad_idx = 0
#     src_batch, trg_batch = [], []
#     for src_sample, trg_sample in batch:
#         src_batch.append(src_sample)
#         trg_batch.append(trg_sample)
    
#     src_batch = pad_sequence(src_batch, padding_value=pad_idx, batch_first=False)
#     trg_batch = pad_sequence(trg_batch, padding_value=pad_idx, batch_first=False)

#     return src_batch, trg_batch


# Create the dataset
dataset = TranslationDataset(train_en, train_vi, SOURCE, TARGET)

BATCH_SIZE = 1
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)#, collate_fn=transpose_batch)
# train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)


ENC_EMB_DIM = 256  # Embedding size for the encoder
DEC_EMB_DIM = 256  # Embedding size for the decoder
HID_DIM = 512      # Hidden dimension size
N_LAYERS = 2       # Number of LSTM layers
DROPOUT = 0.4  # Dropout rate for the encoder
NUM_EPOCHS = 1
LEARNING_RATE = 0.0001  
LIMIT = 20000

encoder = Encoder(input_dim=len(vocab_en_obj), emb_dim=ENC_EMB_DIM, 
                  hid_dim=HID_DIM, n_layers=N_LAYERS, 
                  dropout=DROPOUT).to(device)

decoder = Decoder(output_dim=len(vocab_vi_obj), emb_dim=DEC_EMB_DIM, 
                  hid_dim=HID_DIM, n_layers=N_LAYERS, 
                  dropout=DROPOUT).to(device)


# Define optimizer and criterion
optimizer = optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# for param in encoder.parameters():
#     param.requires_grad = True
# for param in decoder.parameters():
#     param.requires_grad = True


encoder.train()
decoder.train()
# Training loop
for epoch in range(NUM_EPOCHS):
    total_loss = []
    for i, batch in enumerate(train_loader): #tqdm(enumerate(train_loader), desc=f'Epoch {epoch}', total=LIMIT):
        src = torch.transpose(batch['src'], 0, 1).to(device)
        trg = torch.transpose(batch["trg"], 0, 1).to(device)
        optimizer.zero_grad()        
        if src.shape[0] <= 1 or trg.shape[0] <= 1:
            continue
        hidden, cell = encoder(src)
        trg_len = trg.shape[0]
        # batch_size = trg.shape[1]
        outputs = torch.zeros(trg_len, BATCH_SIZE, decoder.output_dim).to(device)
        input = trg[0,:]
        
        for t in range(1, trg_len):
            output, hidden, cell = decoder(input, hidden, cell)
            outputs[t] = output
            input = trg[t]  # Teacher forcing: next input is current target

        # Calculate loss
        output_dim = outputs.shape[-1]
        outputs = outputs[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(outputs, trg)
        total_loss.append(loss.item())
        # print(loss.item())

        # Backward and optimize
        try:
            loss.backward()
        except:
            pass
        # loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)
        optimizer.step()

        if i % 2000 == 0:
            print(f'iteration {i}: loss = {np.mean(total_loss)}')
            total_loss = []

        if i >= LIMIT:
            break
