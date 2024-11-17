# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py
import pandas as pd
import numpy as np
import random
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from .process import annotate

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_c4_unclipped(nsamples, seed, seqlen, tokenizer, strat = "mid"):
    # Load train and validation datasets
    try:
        data_df = pd.read_json('../assets/dataset/c4_std_annotated_picked.jsonl', lines=True)
    except:
        annotate('../assets/dataset/c4_std.jsonl', seqlen, clip = False)
        data_df = pd.read_json('../assets/dataset/c4_std_annotated_noclip.jsonl', lines=True)

    val_df = pd.read_json('../assets/dataset/c4_val.jsonl', lines=True)

    data = Dataset.from_pandas(data_df)
    valdata = Dataset.from_pandas(val_df)
    n = len(data)

    if strat == "mid":
        sorted_data = sorted(data, key=lambda x: x["value_rank"])
        sorted_data = sorted_data[n // 3 + 1: 2 * n // 3]
    elif strat == "low":
        sorted_data = sorted(data, key=lambda x: x["value_rank"])
        sorted_data = sorted_data[:n // 3]
    elif strat == "high":
        sorted_data = sorted(data, key=lambda x: x["value_rank"])
        sorted_data = sorted_data[5 * n // 6 + 1:]    
    else:
        sorted_data = data
    # Generate samples from training set
    print(len(sorted_data))
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(sorted_data) - 1)
            trainenc = tokenizer(sorted_data[i]['text'], return_tensors='pt')
            print(trainenc.input_ids.shape[1] , seqlen)
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

def get_c4_sorted(nsamples, seed, seqlen, tokenizer, strat = "mid"):
    try:
        data_df = pd.read_json('../assets/dataset/c4_large_annotated_picked.jsonl', lines=True)
    except:
        annotate('../assets/dataset/c4_large.jsonl', seqlen)
        data_df = pd.read_json('../assets/dataset/c4_large_annotated_picked.jsonl', lines=True)

    val_df = pd.read_json('../assets/dataset/c4_val.jsonl', lines=True)

    data = Dataset.from_pandas(data_df)
    valdata = Dataset.from_pandas(val_df)
    n = len(data)

    if strat == "mid_no_random":
        sorted_data = sorted(data, key=lambda x: x["diff_rank"])
    elif strat == "low_no_random":
        sorted_data = sorted(data, key=lambda x: x["value_rank"])
    elif strat == "high_no_random":
        sorted_data = sorted(data, key=lambda x: -x["value_rank"])
    elif strat == "mid":
        sorted_data = sorted(data, key=lambda x: x["value_rank"])
        sorted_data = sorted_data[n // 3 + 1: 2 * n // 3]
    elif strat == "low":
        sorted_data = sorted(data, key=lambda x: x["value_rank"])
        sorted_data = sorted_data[:n // 3]
    elif strat == "high":
        sorted_data = sorted(data, key=lambda x: x["value_rank"])
        sorted_data = sorted_data[11 * n // 12 + 1:]    
    else:
        sorted_data = data
    
    random.seed(seed)
    trainloader = []
    token_losses = []
    print(len(sorted_data))
    for _ in tqdm(range(nsamples)):
        token_loss = None
        range_i, range_j = None, None
        while True:
            i = random.randint(0, len(sorted_data) - 1)
            trainenc = tokenizer(sorted_data[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                token_loss = sorted_data[i]['token_losses']
                range_i, range_j = sorted_data[i]['range'][0], sorted_data[i]['range'][1]
                break
        inp = trainenc.input_ids[:, range_i:range_j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
        token_loss = token_loss[0]
        #print(len(token_loss), inp.shape[1], range_i, range_j)
        assert len(token_loss) >= 1
        token_losses.append(token_loss)
    
    #token_losses = np.array(token_losses)
    bar = np.percentile(token_losses, 25)
    token_loss_mask = token_losses > bar
    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc, token_loss_mask

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4_sorted_mid" in name:
        return get_c4_sorted(nsamples, seed, seqlen, tokenizer)    
    if "c4_sorted_low" in name:
        return get_c4_sorted(nsamples, seed, seqlen, tokenizer, strat="low")
    if "c4_sorted_high" in name:
        return get_c4_sorted(nsamples, seed, seqlen, tokenizer, strat="high")    
    if "c4_unclipped_mid" in name:
        return get_c4_unclipped(nsamples, seed, seqlen, tokenizer)    
    if "c4_unclipped_low" in name:
        return get_c4_unclipped(nsamples, seed, seqlen, tokenizer, strat="low")
    if "c4_unclipped_high" in name:
        return get_c4_unclipped(nsamples, seed, seqlen, tokenizer, strat="high")   
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)