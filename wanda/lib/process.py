from .loss_model import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, torch, random
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator

def create_prompt(jsonl_path, seqlen, loss_model = None, clip = True, input_path = "assets/dataset/input"):
    if not loss_model:
        loss_model = '../assets/models/Llama-2-7b-hf'
    tokenizer = AutoTokenizer.from_pretrained(loss_model.config.name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # 逐个样本计算 loss
    sample_losses = []
    for sample in tqdm(data, desc="Calculating losses"):
        # 对每个样本的文本进行编码
        torch.cuda.empty_cache()
        inputs = tokenizer(sample['text'], return_tensors="pt", padding=True, max_length = seqlen+1, truncation=True)

        i = 0
        j = inputs["input_ids"].shape[1] 

        if clip and inputs.input_ids.shape[1] - seqlen > 1:
            i = random.randint(0, inputs.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen

        sample["input_ids"] = inputs["input_ids"][:, i:j]
        sample["attention_mask"] = inputs["attention_mask"][:, i:j]
        sample["labels"] = inputs["input_ids"]
        sample['range'] = [i, j]

    with open(input_path + jsonl_path.replace('.jsonl', '_tokenized.jsonl'), 'w') as f:
        for sample in data:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Annotated JSONL file saved to {input_path + jsonl_path.replace('.jsonl', '_tokenized.jsonl')}")
    
def inference(input_path, loss_model = None, clip = True):
    acc = Accelerator()
    if not loss_model:
        loss_model = '../assets/models/Llama-2-7b-hf'
    model = AutoModelForCausalLM.from_pretrained(loss_model)

def annotate(jsonl_path, seqlen, tokenizer = None, loss_model = None, clip = True):
    '''
    Parrallelization to be implemented. This implementation is conducted on a sample-by-sample basis, for sake of saving information.
    '''
    if torch.cuda.is_available():
        print(f"可见设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
    if not loss_model:
       loss_model = AutoModelForCausalLM.from_pretrained('../assets/models/Llama-2-7b-hf', device_map = "auto", offload_folder="offload", dtype = torch.fp16) 
       #loss_model = AutoModelForCausalLM.from_pretrained('../assets/models/Llama-2-7b-hf', device_map = "auto") 
    if not tokenizer:
       tokenizer = AutoTokenizer.from_pretrained(loss_model.config.name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    print("loss model and loss tokenizer: ", loss_model.name_or_path, tokenizer.name_or_path)
    # 读取 jsonl 文件
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    sample_losses = []
    for sample in tqdm(data, desc="Calculating losses"):
        # 对每个样本的文本进行编码
        torch.cuda.empty_cache()
        inputs = tokenizer(sample['text'], return_tensors="pt", padding=False, max_length = 4096, truncation=True)
        
        if clip:
            if inputs.input_ids.shape[1] - seqlen < 1:
                i = 0
                j = inputs["input_ids"].shape[1]
            else:
                i = random.randint(0, inputs.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
        else:
            i = 0
            j = inputs["input_ids"].shape[1]

        sample['range'] = [i, j]
        
        inputs["input_ids"] = inputs["input_ids"][:, i:j]
        inputs["attention_mask"] = inputs["attention_mask"][:, i:j]
        inputs["labels"] = inputs["input_ids"]

        with torch.no_grad():
            outputs = loss_model(input_ids = inputs["input_ids"], attention_mask = inputs["attention_mask"], labels = inputs["labels"])
            if inputs["input_ids"].shape[1] < seqlen:
                outputs_with_sample_loss = add_sample_loss(outputs, inputs)
                sample_loss = outputs_with_sample_loss["sample_losses"].item()  # 获取样本的平均 loss
                #sample_loss = outputs.loss.item()
                sample['token_losses'] = []
            else:    
                print("long sample", sample["range"])
                outputs_with_sample_loss = add_sample_loss(outputs, inputs)
                sample_loss = outputs_with_sample_loss["sample_losses"].item()  # 获取样本的平均 loss
                #sample_loss = outputs.loss.item()
                sample['token_losses'] = outputs_with_sample_loss['token_losses'].tolist()

            sample_losses.append(sample_loss)
        
        # 将 sample loss 添加到数据中
        sample['sample_loss'] = sample_loss
    
    # 按 sample loss 排序
    sorted_losses = sorted(sample_losses)
    assert sorted_losses[0] < sorted_losses[50]
    
    mean_loss = np.mean(sorted_losses)
    sorted_by_diff = sorted(sample_losses, key=lambda x: abs(x - mean_loss))
    
    # 为每个样本添加排名
    for sample in data:
        try:
            sample['value_rank'] = sorted_losses.index(sample['sample_loss']) + 1  # 获取排序排名
            sample['diff_rank'] = sorted_by_diff.index(sample['sample_loss']) + 1
            sample["diff"] = abs(sample['sample_loss'] - mean_loss)
        except:
            sample['value_rank'] = -1
            sample['diff_rank'] = -1 
            sample['diff'] = -1 
    # 将结果写回 jsonl 文件
    if clip:
        output_path = jsonl_path.replace('.jsonl', '_annotated_picked.jsonl')
    else:
        output_path = jsonl_path.replace('.jsonl', '_annotated.jsonl')
    with open(output_path, 'w') as f:
        for sample in data:
            if sample['value_rank'] == -1:
                print("sample not found")
                continue
            f.write(json.dumps(sample) + '\n')
    
    print(f"Annotated JSONL file saved to {output_path}")

def rearrange(jsonl_path):
    sample_losses = []
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            sample_losses.append(entry["sample_loss"])
            data.append(entry)
    sorted_losses = sorted(sample_losses)
    for sample in data:
        sample['value_rank'] = sorted_losses.index(sample['sample_loss']) + 1  # 获取排序排名
    output_path = jsonl_path.replace('.jsonl', '_annotated_picked.jsonl') 
    with open(output_path, 'w') as f:
        for sample in data:
            f.write(json.dumps(sample) + '\n')
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # m = AutoModelForCausalLM.from_pretrained('baffo32/decapoda-research-llama-7B-hf', device_map = "auto")
    # m = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
    # m = AutoModelForCausalLM.from_pretrained('../assets/models/Llama-2-7b-hf', device_map = "auto")
    # annotate('../assets/dataset/c4_std.jsonl', 2048, m, False)
    rearrange('assets/dataset/c4_large_annotated_picked.jsonl')