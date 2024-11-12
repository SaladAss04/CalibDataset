import torch
from transformers import AutoModelForCausalLM

class CausalLMWithTokenLoss(AutoModelForCausalLM):
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 获取原始输出
        print("ass")
        outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        print(input_ids, attention_mask, labels)
        # 获取 logits 和 labels
        logits = outputs.logits
        if labels is not None:
            # 移动 logits 和 labels 以对齐预测位置
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # 计算逐 token 的损失
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # 将 token loss 重塑为 (batch_size, sequence_length) 形状
            token_losses = token_losses.view(labels.size(0), -1)

            # 计算每个样本的总 loss
            sample_losses = token_losses.mean(dim=1)

            # 将 token loss 添加到模型输出
            print("sample losses: ", sample_losses)
            return {'loss': outputs.loss, 'token_losses': token_losses, 'sample_losses': sample_losses}
        else:
            print("labels not found")
            return outputs

def add_sample_loss(outputs, inputs):
    labels = inputs['labels'] 
    logits = outputs.logits
    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
            # 计算逐 token 的损失
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # 将 token loss 重塑为 (batch_size, sequence_length) 形状
        token_losses = token_losses.view(labels.size(0), -1)
            # 计算每个样本的总 loss
        sample_losses = token_losses.mean(dim=1)

        #print("sample losses: ", sample_losses)
        return {'loss': outputs.loss, 'token_losses': token_losses, 'sample_losses': sample_losses}
    else:
        print("labels not found")
        return outputs