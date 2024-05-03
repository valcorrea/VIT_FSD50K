import torch
from functools import partial
from argparse import ArgumentParser
import time

def main(args):
    fft = partial(torch.fft.fftn, dim=(1,2))
    end = 0
    for i in range(1000):
        batch = torch.rand([args.batch_size, args.seq_len, args.embed_len])
        device = torch.device('cuda')
        batch = batch.to(device)
        start = time.time()
        outputs = fft(batch)
        end += time.time()-start
    print(end/1000)

if __name__ == '__main__':
    ap = ArgumentParser
    ap.add_argument('--seq_len', type=int, help='sequence length')
    ap.add_argument('--embed_len', type=int, help='embedding length')
    ap.add_argument('--batch_size', type=int, help='Batch size')
    args = ap.parse_args()

    main(args)