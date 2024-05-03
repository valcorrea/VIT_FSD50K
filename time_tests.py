import torch
from functools import partial
from argparse import ArgumentParser
import time
from tqdm import tqdm
from src.models.KWT import PreNorm, Attention, FeedForward
from torch import nn

class Fourier(nn.Module):
    def __init__(self):
        self.fourier = partial(torch.fft.fftn, dim=(1,2))
        
    def forward(self, x):
        return self.fourier(x).real

def main(args, fourier_seq, attention_seq):
    end = 0
    batch = torch.rand([int(args.batch_size), int(args.seq_len), int(args.embed_len)])
    device = torch.device('cuda')
    fourier_seq = fourier_seq.to(device)
    attention_seq = attention_seq.to(device)
    batch = batch.to(device)
    fourier_times = []
    attention_times = []
    for i in tqdm(range(1000)):
        start = time.time()
        outputs = fourier_seq(batch)
        end = time.time()-start
        fourier_times.append(end)
        start = time.time()
        outputs = attention_seq(batch)
        end = time.time() - start
        attention_times.append(end)
    fourier_times.pop(0)
    attention_times.pop(0)
    fourier_time = sum(fourier_times)/len(fourier_times)
    attention_time = sum(attention_times)/len(attention_times)
    return fourier_time, attention_time

 
    

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--seq-len', type=str, help='sequence length')
    ap.add_argument('--embed-len', type=str, help='embedding length')
    ap.add_argument('--batch-size', type=str, help='Batch size')
    args = ap.parse_args()

    P_Norm = PreNorm
    attention = P_Norm(args.embed_len, Attention(args.embed_len, heads=3, dim_head=64, dropout=0))
    fourier = P_Norm(args.embed_len, Fourier())
    mlp_attn = P_Norm(args.embed_dim, FeedForward(args.embed_dim, 4 * args.embed_dim, dropout=0))
    mlp_fourier = P_Norm(args.embed_dim, FeedForward(args.embed_dim, 4 * args.embed_dim, dropout=0))
    fourier_seq = nn.Sequential(fourier, mlp_fourier)
    attention_seq = nn.Sequential(attention, mlp_attn)
    
    
    fourier_time, attention_time = main(args, fourier_seq, attention_seq)
    print(f'Fourier_time:{fourier_time}, attention_time{attention_time}')