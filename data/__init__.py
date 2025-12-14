
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
from data.utils import DummyTransform

from data.coco_karpathy_dataset import coco_karpathy_train, coco_karpathy_retrieval_eval
from data.pretrain_dataset import pretrain_dataset


def create_dataset(dataset, config):
    image_processor = AutoImageProcessor.from_pretrained(config['siglip_path'], use_fast=True)
    siglip_transform = lambda x: \
    image_processor(images=x, return_tensors='pt')['pixel_values'][0]
    train_dataset = pretrain_dataset(config['train_file'], siglip_transform)
    return train_dataset


def create_sampler(datasets, shuffles, num_replicas, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_replicas, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = False
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
