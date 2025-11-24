
import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from GraphormerH import GlycanGraphormerEncoder
from ProteinEncoder import ESMCProteinEncoder
from tokenizer import GLYCO_VOCAB_SIZE

class SweetBind(nn.Module):
    """
    Cross-attention fusion of glycan tokens and protein tokens with a glycan encoder
    """
    def __init__(
        self,
        glycan_hidden_dim: int = 256,
        glycan_layers: int = 6,
        protein_dim: int = 1152,  # ESM-C token embedding dim
        dropout: float = 0.1,
        model_type: str = "attention",
        coattention: bool = False,
        heads: int = 8
    ):
        super().__init__()
        self.glycan_hidden_dim = int(glycan_hidden_dim)
        self.protein_dim = int(protein_dim)
        self.dropout = float(dropout)
        self.protein_max_len = 4096
        self.coattention = coattention
        self.device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = torch.float16
        self.ln1 = nn.LayerNorm(self.glycan_hidden_dim)
        self.ln2 = nn.LayerNorm(self.protein_dim)
        self.protein_encoder = ESMCProteinEncoder(cache_path="./data/protein_embeddings.pt")

        self.glycan_encoder = GlycanGraphormerEncoder(
            num_node_types=GLYCO_VOCAB_SIZE,
            num_edge_types=GLYCO_VOCAB_SIZE,
            max_in_degree=32,
            max_out_degree=32,
            max_spatial_distance=32,
            multi_hop_max_dist=5,
            num_layers=glycan_layers,
            hidden_dim=glycan_hidden_dim,
            ffn_dim=glycan_hidden_dim * 2,
            num_heads=heads,
            dropout=0.1,
            attention_dropout=0.1,
        ).to(self.device_type)

        self.graph_ln = nn.LayerNorm(self.glycan_hidden_dim)
        
        self.attention = model_type.lower() == "attention"
        if self.attention:  
            self.g2p = CrossAttention(self.glycan_hidden_dim, self.protein_dim, heads=heads)
        if self.coattention:
            self.p2g = CrossAttention(self.protein_dim, self.glycan_hidden_dim, heads=heads)
            fusion_in_dim = self.glycan_hidden_dim + self.protein_dim
        elif self.attention:
            fusion_in_dim = self.glycan_hidden_dim
        else:
            fusion_in_dim = self.glycan_hidden_dim + self.protein_dim
        self.fusion_block = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_in_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.regression_head = nn.Linear(fusion_in_dim, 1)

    def toggle_glycan_encoder_finetune(self, finetune: bool = True):
        if finetune:
            self.glycan_encoder.train()
        else:
            self.glycan_encoder.eval()
        for param in self.glycan_encoder.parameters():
            param.requires_grad = finetune
        mode = "Fine-tuning" if finetune else "Freezing"
        print(f"[Glycan Encoder] {mode} parameters.")

    def forward(self, glycan_iupac_list: List[str], protein_seq_list: List[str]):
        with torch.autocast(device_type=("cuda" if self.device_type.type == "cuda" else "cpu"), dtype=self.amp_dtype, enabled=(self.device_type.type == "cuda")):
            p_pad, p_mask = self.protein_encoder(protein_seq_list)
            g_pad, g_mask, g_graph = self.glycan_encoder(glycan_iupac_list)

            # Cross attention both ways
            if self.attention:
                g_pad = self.g2p(g_pad, p_pad, g_mask, p_mask)

            g_vec = (g_pad * g_mask.unsqueeze(-1)).sum(1) / g_mask.sum(1, keepdim=True).clamp_min(1)
            g_vec = self.ln1(g_vec)

            if self.coattention:
                p_pad = self.p2g(p_pad, g_pad, p_mask, g_mask)

            if not self.attention or self.coattention:
                p_vec = (p_pad * p_mask.unsqueeze(-1)).sum(1) / p_mask.sum(1, keepdim=True).clamp_min(1)
                p_vec = self.ln2(p_vec)
                fusion_input = torch.cat([g_vec, p_vec], dim=-1)
            else:
                fusion_input = g_vec

            fused = self.fusion_block(fusion_input)
            out = self.regression_head(fused)
            return out.squeeze(-1)

class CrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.heads = heads
        dim_head = q_dim // heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(self.q_dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(self.kv_dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(self.kv_dim, heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, self.q_dim)
        self.layer_norm = nn.LayerNorm(self.q_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, glycan, protein, glycan_mask, pro_mask):
        b, n, _, h = *glycan.shape, self.heads
        
        # Project glycan into query space
        q = self.to_q(glycan).view(b, n, self.heads, -1).transpose(1, 2)
        
        # Project protein into key and value space
        target_len = protein.shape[1]
        k = self.to_k(protein).view(b, target_len, self.heads, -1).transpose(1, 2)
        v = self.to_v(protein).view(b, target_len, self.heads, -1).transpose(1, 2)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        glycan_mask = glycan_mask.unsqueeze(1).unsqueeze(-1).expand(-1, self.heads, n, target_len)
        masked_dots = dots.masked_fill(glycan_mask == 0, -1e4)
        pro_mask = pro_mask.unsqueeze(1).unsqueeze(-2).expand(-1, self.heads, n, target_len)
        masked_dots = masked_dots.masked_fill(pro_mask == 0, -1e4)
        
        # Apply softmax to compute attention weights
        attn = F.softmax(masked_dots, dim=-1)
        attn = self.dropout(attn)

        # Compute output
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)

        out = self.to_out(out)
        
        assert glycan.shape == out.shape, "Shape mismatch between glycan and out"
        out = self.layer_norm(out + glycan)
            
        return out