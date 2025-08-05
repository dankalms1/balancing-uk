from .datasets import LSTMDataset, MultiFeedDataset

DATASET_FACTORY = {
    "LSTMDataset": LSTMDataset,
    "MultiFeedDataset": MultiFeedDataset
}