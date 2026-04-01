"""Small dataset wrappers for inference-generation task execution."""

from __future__ import annotations

from torch.utils.data import Dataset


class ListTaskDataset(Dataset):
    """Expose a list of task dictionaries through the Dataset interface."""

    def __init__(self, tasks: list[dict]):
        self.tasks = list(tasks)

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> dict:
        return self.tasks[index]


def task_collate_fn(batch: list[dict]) -> list[dict]:
    """Keep task dictionaries intact during DataLoader collation."""
    return batch
