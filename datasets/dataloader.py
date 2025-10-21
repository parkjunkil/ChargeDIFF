import torch.utils.data

from datasets.base_dataset import CreateDataset
from datasets.base_dataset import data_sampler

from torch_geometric.data import Batch
from torch.utils.data import DataLoader

def get_data_generator(loader):
    while True:
        for data in loader:
            yield data


def custom_collate_fn(batch):
    
    # Convert each to tensor
    chgden_batch = [d['chgden'] for d in batch]
    struc_batch  = [d['struc'] for d in batch]
    prop_batch = [d['prop'] for d in batch]
                
    chgden_batch = torch.stack(chgden_batch)
    struc_batch = Batch.from_data_list(struc_batch)
    
    if not any(element is None for element in prop_batch): #for unconditional
        prop_batch = torch.stack(prop_batch)
        
    return {
        'chgden': chgden_batch,
        'struc': struc_batch,
        'prop': prop_batch
    }


def CreateDataLoader(opt):
    train_dataset, val_dataset, test_dataset = CreateDataset(opt)

    train_dl = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            sampler=data_sampler(train_dataset, shuffle=True, distributed=opt.distributed),
            drop_last=True,
            num_workers = 8,
            pin_memory=True,
            collate_fn=custom_collate_fn
            )

    val_dl= DataLoader(
            val_dataset,
            #batch_size=max(int(opt.batch_size // 2), 1),
            batch_size=opt.batch_size,
            sampler=data_sampler(val_dataset, shuffle=False, distributed=opt.distributed),
            drop_last=False,
            num_workers = 8,
            pin_memory=True,
            collate_fn=custom_collate_fn
            )

    test_dl = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            sampler=data_sampler(test_dataset, shuffle=False, distributed=opt.distributed),
            drop_last=False,
            num_workers = 8,
            pin_memory=True,
            collate_fn=custom_collate_fn
            )

    return train_dl, val_dl, test_dl
