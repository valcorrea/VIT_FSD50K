from tqdm import tqdm
from glob import glob
import os
import soundfile as sf

def main(root_dir):
    for file in tqdm(glob(os.path.join(root_dir, '*.wav'))):
        y, _ = sf.read(file, dtype='int16')
        if max(abs(y)) < 1500:
            os.remove(file)

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('root_dir', type=str, help='Directory containing wav files.')
    args = ap.parse_args()

    main(args.root_dir)