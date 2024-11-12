from loss_model import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, torch, random
from tqdm import tqdm
import numpy as np
def annotate(jsonl_path, seqlen, loss_model = None, clip = True):
    # 加载 tokenizer
    if not loss_model:
       loss_model = AutoModelForCausalLM.from_pretrained('../assets/models/Llama-2-7b-hf', device_map = "auto") 
    tokenizer = AutoTokenizer.from_pretrained(loss_model.config.name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 读取 jsonl 文件
    with open(jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # 逐个样本计算 loss
    sample_losses = []
    for sample in tqdm(data, desc="Calculating losses"):
        # 对每个样本的文本进行编码
        torch.cuda.empty_cache()
        inputs = tokenizer(sample['text'], return_tensors="pt", padding=True, max_length = 7000, truncation=True)
        
        if clip:
            if inputs.input_ids.shape[1] - seqlen - 1 < 1:
                continue
            i = random.randint(0, inputs.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
        else:
            i = 0
            j = inputs["input_ids"].shape[1]

        sample['range'] = [i, j]
        
        inputs["input_ids"] = inputs["input_ids"][:, i:j]
        inputs["attention_mask"] = inputs["attention_mask"][:, i:j]
        inputs["labels"] = inputs["input_ids"]

        # 前向传播计算 loss
        with torch.no_grad():
            #outputs = loss_model(input_ids = inputs["input_ids"], attention_mask = inputs["attention_mask"], labels = inputs["labels"])
            outputs = loss_model(input_ids = inputs["input_ids"], attention_mask = inputs["attention_mask"], labels = inputs["labels"])
            outputs_with_sample_loss = add_sample_loss(outputs, inputs)
            sample_loss = outputs_with_sample_loss["sample_losses"].item()  # 获取样本的平均 loss
            sample_losses.append(sample_loss)
        
        # 将 sample loss 添加到数据中
        sample['sample_loss'] = sample_loss
        sample['token_losses'] = outputs_with_sample_loss['token_losses'].tolist()
    
    # 按 sample loss 排序
    sorted_losses = sorted(sample_losses, reverse=True)
    
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
                continue
            f.write(json.dumps(sample) + '\n')
    
    print(f"Annotated JSONL file saved to {output_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # m = AutoModelForCausalLM.from_pretrained('baffo32/decapoda-research-llama-7B-hf', device_map = "auto")
    # m = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
    m = AutoModelForCausalLM.from_pretrained('../assets/models/Llama-2-7b-hf', device_map = "auto")
    annotate('../assets/dataset/c4_std.jsonl', 2048, m, False)