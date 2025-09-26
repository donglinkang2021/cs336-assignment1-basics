import torch
import numpy as np
import numpy.typing as npt

# uv run pytest -k test_get_batch
def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. Generate random starting indices for the batches
    start_indices = np.random.randint(0, len(dataset) - context_length, size=(batch_size,))

    # 2. Create indices for contiguous blocks of size context_length + 1
    offsets = np.arange(context_length + 1)
    block_indices = start_indices[:, None] + offsets

    # 3. Index the dataset once to get all data blocks
    data_blocks = torch.from_numpy(dataset[block_indices].astype(np.int64))

    # 4. Create x and y by slicing the data blocks. This is a very fast operation.
    x = data_blocks[:, :-1]
    y = data_blocks[:, 1:]

    # 5. Move tensors to the specified device
    x, y = x.to(device), y.to(device)
    return x, y
