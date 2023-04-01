import json
import torch
import torch.utils.data as data
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils.config import *
import logging 
import datetime
import ast

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

MEM_TOKEN_SIZE = 5

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {UNK_token: 'UNK', PAD_token: "PAD", EOS_token: "EOS",  SOS_token: "SOS"}
        self.n_words = 4 # Count default tokens
      
    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, src_seq, trg_seq, index_seq, gate_seq,src_word2id, trg_word2id,max_len,entity,entity_cal,entity_nav,entity_wet, conv_seq, kb_arr):
        """Reads source and target sequences from txt files."""
        self.src_seqs = src_seq
        self.trg_seqs = trg_seq
        self.index_seqs = index_seq   
        self.gate_seq = gate_seq     
        self.num_total_seqs = len(self.src_seqs)
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.max_len = max_len
        self.entity = entity
        self.entity_cal = entity_cal
        self.entity_nav = entity_nav
        self.entity_wet = entity_wet
        self.conv_seq = conv_seq
        self.kb_arr = kb_arr
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        trg_seq = self.trg_seqs[index]
        index_s = self.index_seqs[index]
        gete_s  = self.gate_seq[index]
        src_seq = self.preprocess(src_seq, self.src_word2id, trg=False)
        trg_seq = self.preprocess(trg_seq, self.trg_word2id)
        index_s = self.preprocess_inde(index_s,src_seq)
        gete_s  = self.preprocess_gate(gete_s)
        conv_seq = self.conv_seq[index]
        conv_seq = self.preprocess(conv_seq, self.src_word2id, trg=False)
        
        return src_seq, trg_seq, index_s, gete_s,self.max_len,self.src_seqs[index],self.trg_seqs[index],self.entity[index],self.entity_cal[index],self.entity_nav[index],self.entity_wet[index], conv_seq, self.kb_arr[index]

    def __len__(self):
        return self.num_total_seqs
    
    def preprocess(self, sequence, word2id, trg=True):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')]+ [EOS_token]
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for word in word_triple:
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        try:
            story = torch.Tensor(story)
        except:
            print(sequence)
            print(story)
        return story

    def preprocess_inde(self, sequence, src_seq):
        """Converts words to ids."""
        sequence = sequence + [len(src_seq)-1]
        sequence = torch.Tensor(sequence)
        return sequence

    def preprocess_gate(self, sequence):
        """Converts words to ids."""
        sequence = sequence + [0]
        return torch.Tensor(sequence)

def collate_fn(data):
    def merge(sequences,max_len):
        lengths = [len(seq) for seq in sequences]
        if (max_len):
            padded_seqs = torch.ones(len(sequences), max(lengths), MEM_TOKEN_SIZE).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i,:end,:] = seq[:end]
        else:
            padded_seqs = torch.ones(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[-1]), reverse=True)
    # seperate source and target sequences
    src_seqs, trg_seqs, ind_seqs, gete_s, max_len, src_plain,trg_plain, entity,entity_cal,entity_nav,entity_wet, conv_seq, kb_arr = zip(*data)
    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs,max_len)
    trg_seqs, trg_lengths = merge(trg_seqs,None)
    ind_seqs, _ = merge(ind_seqs,None)
    gete_s, _ = merge(gete_s,None)
    conv_seqs, conv_lengths = merge(conv_seq, max_len)
    
    src_seqs = Variable(src_seqs).transpose(0,1)
    trg_seqs = Variable(trg_seqs).transpose(0,1)
    ind_seqs = Variable(ind_seqs).transpose(0,1)
    gete_s = Variable(gete_s).transpose(0,1)
    conv_seqs = Variable(conv_seqs).transpose(0,1)

    if USE_CUDA:
        src_seqs = src_seqs.cuda()
        trg_seqs = trg_seqs.cuda()
        ind_seqs = ind_seqs.cuda()
        gete_s = gete_s.cuda()
        conv_seqs = conv_seqs.cuda()
    return src_seqs, src_lengths, trg_seqs, trg_lengths, ind_seqs, gete_s, src_plain, trg_plain, entity, entity_cal, entity_nav, entity_wet, conv_seqs, conv_lengths, kb_arr


def read_langs(file_name, max_line = None):
    logging.info(f"Reading lines from {file_name}")
    data=[]
    contex_arr = []
    conversation_arr = []
    kb_arr = []
    entity = {}
    u=None
    r=None
    with open(file_name) as fin:
        cnt_ptr = 0
        cnt_voc = 0
        max_r_len = 0
        cnt_lin = 1
        user_counter = 0
        system_counter = 0
        system_res_counter = 0
        KB_counter = 0
        dialog_counter = 0
        for line in fin:
            if line := line.strip():
                if '#' in line:
                    line = line.replace("#","")
                    task_type = line
                    continue
                nid, line = line.split(' ', 1)
                if '\t' in line:
                    u, r, gold = line.split('\t')
                    user_counter += 1
                    system_counter += 1

                    gen_u = generate_memory(u, "$u", str(nid))
                    contex_arr += gen_u
                    conversation_arr += gen_u

                    r_index = []
                    gate = []
                    for key in r.split(' '):
                        index = [loc for loc, val in enumerate(contex_arr) if (val[0] == key)]
                        if (index):
                            index = max(index)
                            gate.append(1)
                            cnt_ptr +=1
                        else: 
                            index = len(contex_arr)
                            gate.append(0)  
                            cnt_voc +=1             
                        r_index.append(index)
                        system_res_counter += 1

                    if len(r_index) > max_r_len: 
                        max_r_len = len(r_index)
                    contex_arr_temp = contex_arr + [['$$$$']*MEM_TOKEN_SIZE]

                    ent_index_calendar = []
                    ent_index_navigation = []
                    ent_index_weather = []

                    gold = ast.literal_eval(gold)
                    if task_type == "navigate":
                        ent_index_navigation = gold

                    elif task_type == "schedule":
                        ent_index_calendar = gold
                    elif task_type == "weather":
                        ent_index_weather = gold
                    ent_index = list(set(ent_index_calendar + ent_index_navigation + ent_index_weather))
                    data.append([contex_arr_temp,r,r_index,gate,ent_index,list(set(ent_index_calendar)),list(set(ent_index_navigation)),list(set(ent_index_weather)), list(conversation_arr), list(kb_arr)])

                    gen_r = generate_memory(r, "$s", str(nid))
                    contex_arr += gen_r
                    conversation_arr += gen_r
                else:
                    KB_counter += 1
                    r=line
                    for e in r.split(' '):
                        entity[e] = 0
                    kb_info = generate_memory(r, "", str(nid))
                    contex_arr += kb_info
                    kb_arr += kb_info
            else:
                cnt_lin+=1
                entity = {}
                if(max_line and cnt_lin>=max_line):
                    break
                contex_arr = []
                conversation_arr = []
                kb_arr = []
                dialog_counter += 1

    max_len = max(len(d[0]) for d in data)
    logging.info(f"Pointer percentace= {cnt_ptr / (cnt_ptr + cnt_voc)} ")
    logging.info(f"Max responce Len: {max_r_len}")
    logging.info(f"Max Input Len: {max_len}")
    logging.info(f"Avg. User Utterances: {user_counter * 1.0 / dialog_counter}")
    logging.info(f"Avg. Bot Utterances: {system_counter * 1.0 / dialog_counter}")
    logging.info(f"Avg. KB results: {KB_counter * 1.0 / dialog_counter}")
    logging.info(f"Avg. responce Len: {system_res_counter * 1.0 / system_counter}")

    print('Sample: ',data[1][0],data[1][1],data[1][2],data[1][3],data[1][4])
    return data, max_len, max_r_len

def generate_memory(sent, speaker, time):
    sent_new = []
    sent_token = sent.split(' ')
    if speaker in ["$u", "$s"]:
        for word in sent_token:
            temp = [word, speaker, f't{str(time)}'] + ["PAD"]*(MEM_TOKEN_SIZE-3)
            sent_new.append(temp)
    else:
        sent_token = sent_token[::-1] + ["PAD"]*(MEM_TOKEN_SIZE-len(sent_token))
        sent_new.append(sent_token)
    return sent_new

def get_seq(pairs,lang,batch_size,type,max_len):   
    x_seq = []
    y_seq = []
    ptr_seq = []
    gate_seq = []
    entity = []
    entity_cal = []
    entity_nav = []
    entity_wet = []
    conv_seq = []
    kb_arr = []

    for pair in pairs:
        x_seq.append(pair[0])
        y_seq.append(pair[1])
        ptr_seq.append(pair[2])
        gate_seq.append(pair[3])
        entity.append(pair[4])
        entity_cal.append(pair[5])
        entity_nav.append(pair[6])
        entity_wet.append(pair[7])
        conv_seq.append(pair[8])
        kb_arr.append(pair[9])
        if(type):
            lang.index_words(pair[0])
            lang.index_words(pair[1], trg=True)

    dataset = Dataset(x_seq, y_seq,ptr_seq,gate_seq,lang.word2index, lang.word2index,max_len,entity,entity_cal,entity_nav,entity_wet, conv_seq, kb_arr)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=type,
        collate_fn=collate_fn,
    )

def prepare_data_seq(task,batch_size=100,shuffle=True):
    file_train = f'data/KVR/{task}train.txt'
    file_dev = f'data/KVR/{task}dev.txt'
    file_test = f'data/KVR/{task}test.txt'

    pair_train,max_len_train, max_r_train = read_langs(file_train, max_line=None)
    pair_dev,max_len_dev, max_r_dev = read_langs(file_dev, max_line=None)
    pair_test,max_len_test, max_r_test = read_langs(file_test, max_line=None)
    max_r_test_OOV = 0
    max_len_test_OOV = 0

    max_len = max(max_len_train,max_len_dev,max_len_test,max_len_test_OOV) +1
    max_r  = max(max_r_train,max_r_dev,max_r_test,max_r_test_OOV) +1
    lang = Lang()

    train = get_seq(pair_train,lang,batch_size,True,max_len)
    dev   = get_seq(pair_dev,lang,batch_size,False,max_len)
    test  = get_seq(pair_test,lang,batch_size,False,max_len)

    logging.info(f"Read {len(pair_train)} sentence pairs train")
    logging.info(f"Read {len(pair_dev)} sentence pairs dev")
    logging.info(f"Read {len(pair_test)} sentence pairs test")
    logging.info(f"Max len Input {max_len} ")
    logging.info(f"Vocab_size {lang.n_words} ")
    logging.info(f"USE_CUDA={USE_CUDA}")
    #print(lang.index2word)

    return train, dev, test, [], lang, max_len, max_r

