import torch
import torch.nn as nn

# Define WrappedGPT class
class WrappedGPT_old:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples

class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none", token_loss_mask=None):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0
        #self.calculated_samples = torch.zeros((self.columns), device = self.dev)

        self.layer_id = layer_id 
        self.layer_name = layer_name

        self.token_loss_mask = token_loss_mask
    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        #print(self.nsamples, self.nsamples + tmp)
        mask = torch.tensor(self.token_loss_mask[self.nsamples:self.nsamples + tmp, :]).view(1, -1)
        mask = torch.cat((torch.zeros(mask.size(0), 1), mask), dim = 1)
        mask = mask.repeat(inp.shape[0], 1)
        
        #self.scaler_row *= self.calculated_samples / (self.calculated_samples + mask.sum())
        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        #self.calculated_samples += mask.sum()

        inp = inp.type(torch.float32)
        inp[mask == False] = 0
        inc = torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples
        inc *= mask.shape[1] / mask[0].sum()
        self.scaler_row += inc