from tokenizers import Tokenizer
import numpy as np
from pathlib import Path
from cs336_basics.data import get_batch

tokenizer_path = "hf_tokenizer/tinystories/tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

data_path = Path("data/tinystories")
train_data = np.memmap(data_path / 'train.bin', dtype=np.uint16, mode='r')

x, y = get_batch(train_data, 4, 256, "cuda")
print(x.shape, y.shape)

print(tokenizer.decode(x[0].tolist()))