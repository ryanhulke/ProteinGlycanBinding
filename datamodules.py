from __future__ import annotations
from typing import Tuple, List
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from utils import get_logger
import os

logg = get_logger()


def raw_pg_collate_fn(batch: List[Tuple[str, str, torch.Tensor]]):
    glycans, proteins, labels = zip(*batch)
    labels = torch.stack(labels, dim=0)
    return list(glycans), list(proteins), labels


class GlycoProteinDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
        glycan_col: str = "Glycan",
        protein_col: str = "Protein",
        label_col: str = "Label"):
        self.df = df.reset_index(drop=True)
        self.glycan_col = glycan_col
        self.protein_col = protein_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[str, str, torch.Tensor]:
        r = self.df.iloc[idx]
        g = str(r[self.glycan_col])
        p = str(r[self.protein_col])
        y = torch.tensor(float(r[self.label_col]), dtype=torch.float32)
        return g, p, y


class GlycoProteinDataModule:
    def __init__(self, path: str, batch_size: int = 32, num_workers: int = 0):
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wide_df = pd.read_csv(self.path)
        # Convert wide matrix with split indicator columns into long-form (glycan, protein, label) for each split
        self.train_df = self.wide_to_long('train')
        self.val_df = self.wide_to_long('valid')
        self.test_df  = self.wide_to_long('test')
        print('Train size:', len(self.train_df), 'Valid size:', len(self.val_df), 'Test size:', len(self.test_df))

    def make_loader(self, df: pd.DataFrame, shuffle: bool) -> DataLoader:
        ds = GlycoProteinDataset(df)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=raw_pg_collate_fn,
            pin_memory=True,
            drop_last=True
        )

    def train_dataloader(self):
        return self.make_loader(self.train_df, shuffle=True)

    def val_dataloader(self):
        return self.make_loader(self.val_df, shuffle=False)

    def test_dataloader(self):
        return self.make_loader(self.test_df, shuffle=False)
    
    def wide_to_long(self, mask_col):
        # Filter rows where the split indicator ==1
        split_cols = [c for c in self.wide_df.columns if c in {'train','valid','test'}]
        protein_col = "target"
        glycan_cols = [c for c in self.wide_df.columns if c not in split_cols + [protein_col]]
        sub = self.wide_df[self.wide_df[mask_col]==1]
        # Melt glycan columns
        melted = sub.melt(id_vars=["target"], value_vars=glycan_cols, var_name='Glycan', value_name='Label')
        # Drop NaN labels
        melted = melted.dropna(subset=['Label']).copy()
        return melted[['Glycan', protein_col, 'Label']].rename(columns={protein_col:'Protein'})
    
class GlycanPretrainDataset(Dataset):
    def __init__(self, csv_path: str):
        super().__init__()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")

        df = pd.read_csv(csv_path)

        # First column is glycan strings
        self.glycan_col = df.columns[0]
        self.glycans: List[str] = df[self.glycan_col].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.glycans)

    def __getitem__(self, idx: int) -> str:
        return self.glycans[idx]

class GlycanPretrainDataModule:
    def __init__(
        self,
        csv_path: str = "./data/glycan_pretrain.csv",
        batch_size: int = 64,
        num_workers: int = 4,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        seed: int = 1,
    ):
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.seed = seed

        self.dataset = GlycanPretrainDataset(csv_path)

        n_total = len(self.dataset)
        n_test = int(n_total * self.test_fraction)
        n_val = int(n_total * self.val_fraction)
        n_train = n_total - n_val - n_test

        generator = torch.Generator().manual_seed(self.seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [n_train, n_val, n_test],
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )