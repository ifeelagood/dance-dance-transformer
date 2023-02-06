import time
import torch
import tqdm
from dataset import OnsetDataset, train_valid_split
from config import config




def time_function(f):
    start = time.time()
    f()
    end = time.time()
    return end - start

def iterate_dataloader(dataloader):
    for features, labels in tqdm.tqdm(dataloader, total=len(dataloader)):
        pass

def single_process(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=0)

    return time_function(lambda: iterate_dataloader(dataloader))

def multi_process(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=config.training.num_workers)

    return time_function(lambda: iterate_dataloader(dataloader))

def benchmark():
    train_manifest, valid_manifest = train_valid_split()

    valid_dataset = OnsetDataset(valid_manifest)

    single_process_time = single_process(valid_dataset)
    multi_process_time = multi_process(valid_dataset)

    print(f"Single process time: {single_process_time:.2f} seconds")
    print(f"Multi process time: ({config.training.num_workers} workers): {multi_process_time:.2f} seconds")


if __name__ == "__main__":
    benchmark()