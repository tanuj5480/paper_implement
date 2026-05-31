import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]



def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = DatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader



if __name__ == "__main__":

    lst = []
    with open("the-verdict.txt", "r", encoding="utf-8") as file:
        content = file.read()
        lst.append(content)
    
    sample_data = lst[:2]
    st = ''
    for elem in lst:
        st += elem

    dataloader = create_dataloader_v1(st, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=1)

    # 1. Convert the DataLoader into a Python Iterator
    data_iter = iter(dataloader)

    # 2. Extract exactly one batch of data
    batch = next(data_iter)

    # 3. If your DataLoader returns features and labels: (images, labels) or (inputs, targets)
    inputs, targets = batch

    # 4. Print shapes and data types
    print("--- Single Batch Inspection ---")
    print(f"Inputs Shape : {inputs.shape}")   # e.g., torch.Size([32, 3, 224, 224])
    print(f"Inputs Type  : {inputs.dtype}")   # e.g., torch.float32
    print(f"Targets Shape: {targets.shape}")  # e.g., torch.Size([32])
    print(f"Targets Type : {targets.dtype}")  # e.g., torch.int64