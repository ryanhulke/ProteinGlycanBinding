from typing import Dict, List, Tuple
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import os
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
import pandas as pd
import re
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

class ESMCProteinEncoder(torch.nn.Module):
    def __init__(self, cache_path: str = None):
        super().__init__()
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = torch.float16 if self.device_type.type == "cuda" else torch.float32
        self.protein_dim = 1152
        self.protein_max_len = 2048
        self.cache_path = cache_path
        self.inmem_protein: Dict[str, torch.Tensor] = {}

        # Protein encoder
        if cache_path is None:
            self.esmc = ESMC.from_pretrained("esmc_600m").to(dtype=self.amp_dtype, device=self.device_type)
            self.esmc_cfg = LogitsConfig(sequence=True, return_embeddings=True)

            # LoRA on protein encoder
            self.init_esmc_lora(top_layers=10, r=8, alpha=32, lora_dropout=0.01)

        # check if the file exists
        if cache_path is not None and not os.path.isfile(cache_path):
            esmc = ESMC.from_pretrained("esmc_600m").to(dtype=torch.float16, device=self.device_type)
            cfg = LogitsConfig(sequence=True, return_embeddings=True)
            out_dict: Dict[str, torch.Tensor] = {}
            seq1 = pd.read_csv("./data/glycan_interaction.csv")["target"].unique().tolist()
            seqs2 = pd.read_csv("./data/LectinOracle_CFG_zscore.csv")["target"].unique().tolist()
            seqs = list(set(seq1 + seqs2))
            print(f"Encoding {len(seqs)} unique protein sequences for cache...")
            for s in tqdm(seqs):
                seq = s if len(s) <= self.protein_max_len else s[:self.protein_max_len]
                toks = esmc.encode(ESMProtein(sequence=seq))
                out = esmc.logits(toks, cfg)
                emb = out.embeddings
                if emb.ndim == 3:
                    emb = emb.squeeze(0)  # [L, D]
                out_dict[s] = emb.detach().to(torch.float16).cpu()  # store as fp16
            meta = {
                "proteins_count": len(out_dict),
                "esmc": "esmc_600m",
            }
            print(meta)
            torch.save({"meta": meta, "embeddings": out_dict}, cache_path)
        data = torch.load(cache_path, map_location="cpu")
        self.inmem_protein = {k: v.to(self.device_type) for k, v in data["embeddings"].items()}

    @torch.no_grad()
    def encode_proteins(self, seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        embs, lens, missing = [], [], []
        for s in seqs:
            key = (s or "").upper().strip()
            t = self.inmem_protein.get(key, None)
            if t is None:
                missing.append(key)
            else:
                e = t.to(device=self.device_type, dtype=self.amp_dtype, non_blocking=True)
                embs.append(e); lens.append(e.size(0))
        if missing:
            raise RuntimeError(f"Protein cache miss for {len(missing)} keys. Example: {missing[0]}")
        p_pad = pad_sequence(embs, batch_first=True, padding_value=0.0)
        p_mask = torch.zeros((len(seqs), p_pad.size(1)), dtype=torch.bool, device=self.device_type)
        for i, L in enumerate(lens):
            p_mask[i, :L] = True
        return p_pad, p_mask

    def forward(self, protein_seq_list: List[str]):
        if self.cache_path is not None:
            p_pad, p_mask = self.encode_proteins(protein_seq_list)
        else:
            p_pad, p_mask = self.encode_proteins_train(protein_seq_list)
        return p_pad, p_mask

    def encode_proteins_train(self, seqs: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        embs, lens = [], []
        for s in seqs:
            seq = (s or "").upper().strip()
            if len(seq) > self.protein_max_len:
                seq = seq[: self.protein_max_len]
            toks = self.esmc.encode(ESMProtein(sequence=seq))
            out = self.esmc.logits(toks, self.esmc_cfg)  # embeddings with grad
            e = out.embeddings
            if e.ndim == 3:
                e = e.squeeze(0)
            embs.append(e.to(device=self.device_type, non_blocking=True))
            lens.append(e.size(0))
        p_pad = pad_sequence(embs, batch_first=True, padding_value=0.0)
        p_mask = torch.zeros((len(seqs), p_pad.size(1)), dtype=torch.bool, device=self.device_type)
        for i, L in enumerate(lens):
            p_mask[i, :L] = True
        return p_pad, p_mask
    
    def init_esmc_lora(self, *, top_layers: int = 4, r: int = 8, alpha: int = 32, lora_dropout: float = 0.05):
        block_idxs = set()
        for name, _ in self.esmc.named_modules():
            m = re.search(r"transformer\.blocks\.(\d+)\.", name)
            if m:
                block_idxs.add(int(m.group(1)))
        max_idx = max(block_idxs)
        start_idx = max(0, max_idx - top_layers + 1)

        targets = []
        for name, module in self.esmc.named_modules():
            if not isinstance(module, (nn.Linear, nn.MultiheadAttention)):
                continue
            m = re.search(r"transformer\.blocks\.(\d+)\.", name)
            if not m:
                continue
            i = int(m.group(1))
            if i < start_idx:
                continue
            if (
                ".attn.layernorm_qkv.0" in name
                or ".attn.layernorm_qkv.1" in name
                or ".attn.out_proj" in name
                or ".ffn.0" in name
                or ".ffn.3" in name
            ):
                targets.append(name)

        lora_cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=lora_dropout,
            target_modules=targets, bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.esmc = get_peft_model(self.esmc, lora_cfg)

        trainable = sum(p.numel() for p in self.esmc.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.esmc.parameters())
        print(f"[LoRA] Injected into {len(targets)} modules across blocks {start_idx}..{max_idx}. "
              f"Trainable {trainable}/{total} params.")