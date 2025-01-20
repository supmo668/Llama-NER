import torch
from torch.utils.data import DataLoader, default_collate


def custom_collate(batch, device="cuda"):
    # Convert each field in the batch to a tensor and move to the specified device
    for i in range(len(batch)):
        for key in batch[i]:
            if isinstance(batch[i][key], list):
                batch[i][key] = torch.tensor(batch[i][key])
            else:
                batch[i][key] = batch[i][key]
    return default_collate(batch)


def get_dataloader(dataset, batch_size, num_workers=3, persistent_workers=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=custom_collate
    )
