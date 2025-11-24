import math
from typing import Optional, Dict, Tuple, List
import torch
from torch import nn
import torch.nn.functional as F
from rdkit import Chem
from tokenizer import GlycanGraph, GLYCO_VOCAB_SIZE, GLYCOWORDS, iupac_to_glycan_graph

Batch = Dict[str, torch.Tensor]

def init_params(module: nn.Module, n_layers: int) -> None:
    """Graphormer style init for small modules."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def init_graphormer_params(module: nn.Module) -> None:
    """Init used for the core Graphormer encoder."""

    def normal_(data: torch.Tensor) -> None:
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if getattr(module, "padding_idx", None) is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GraphNodeFeature(nn.Module):
    """Node type + degree encodings + graph token."""

    def __init__(
        self,
        num_heads: int,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        hidden_dim: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_atoms = num_atoms

        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)

        self.apply(lambda m: init_params(m, n_layers=n_layers))

    def forward(self, batched: Batch) -> torch.Tensor:
        """
        Args:
            batched["x"]: [B, N, 1] or [B, N, feat_dim]
        Returns:
            [B, N+1, H] including graph token at index 0
        """
        x = batched["x"]
        in_degree = batched["in_degree"]
        out_degree = batched["out_degree"]

        if x.dim() == 3:
            if x.size(-1) == 1:
                node_feature = self.atom_encoder(x.squeeze(-1))  # [B, N, H]
            else:
                node_feature = self.atom_encoder(x).sum(dim=-2)  # [B, N, H]
        else:
            node_feature = self.atom_encoder(x)                  # [B, N, H]

        node_feature = (
            node_feature
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )

        B, N, H = node_feature.shape
        graph_token = self.graph_token.weight.view(1, 1, H).expand(B, 1, H)
        return torch.cat([graph_token, node_feature], dim=1)  # [B, N+1, H]


class GraphAttnBias(nn.Module):
    """
    Compute attention bias per head, including full multi-hop edge encoding as in Graphormer.

    With a simple multi-channel mechanism:
      - A subset of heads are "local" heads that only attend within a radius
        in shortest path distance (SPD).
      - Remaining heads are "global" heads that can still attend everywhere.

    Expects in batched_data:
        attn_bias:      [B, N+1, N+1]
        spatial_pos:    [B, N, N]       shortest-path distances (0 = self)
        attn_edge_type: [B, N, N, E]    (used only if edge_type != 'multi_hop')
        edge_input:     [B, N, N, max_dist, 1]  hop-wise edge types along path
        x:              [B, N, ...]
    """

    def __init__(
        self,
        num_heads: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        n_layers: int,
        # new multi-channel params
        use_multi_channel: bool = False,
        num_local_heads: Optional[int] = None,
        local_radius: int = 1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.multi_hop_max_dist = multi_hop_max_dist
        self.edge_type = edge_type

        # multi-channel config
        self.use_multi_channel = use_multi_channel
        self.local_radius = local_radius
        if use_multi_channel:
            if num_local_heads is None:
                num_local_heads = num_heads // 2
            if not (1 <= num_local_heads <= num_heads):
                raise ValueError("num_local_heads must be in [1, num_heads]")
        self.num_local_heads = num_local_heads

        # Edge type -> per head bias
        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)

        if self.edge_type == "multi_hop":
            # distance-dependent mixing as in Graphormer
            self.edge_dis_encoder = nn.Embedding(
                num_edge_dis * num_heads * num_heads,
                1,
            )

        # SPD embedding (0..num_spatial-1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)

        # Graph-token "virtual distance"
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.apply(lambda m: init_params(m, n_layers=n_layers))

    def forward(self, batched_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        attn_bias = batched_data["attn_bias"]  # [B, N+1, N+1]
        spatial_pos = batched_data["spatial_pos"]  # [B, N, N]
        x = batched_data["x"]  # [B, N, ...]
        edge_input = batched_data["edge_input"]  # [B, N, N, max_dist, 1]
        attn_edge_type = batched_data["attn_edge_type"]  # [B, N, N, 1]

        B, N = x.size()[:2]

        # Base bias → [B, H, N+1, N+1]
        graph_attn_bias = attn_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # === Spatial encoding (SPD) ===
        # [B,N,N,H] ->[B,H,N,N]
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] += spatial_pos_bias

        # === Graph-token virtual distance ===
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] += t
        graph_attn_bias[:, :, 0, :] += t

        # === Edge encoding ===
        if self.edge_type == "multi_hop":
            # spatial_pos_ used for normalization, following Graphormer
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1           # avoid div by zero
            spatial_pos_ = torch.where(
                spatial_pos_ > 1,
                spatial_pos_ - 1,
                spatial_pos_,
            )

            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]

            # edge_input: [B,N,N,max_dist,H] (H = num_heads)
            hop_emb = self.edge_encoder(edge_input).mean(-2)  # mean over last dim
            # hop_emb: [B,N,N,max_dist,num_heads]

            max_dist = hop_emb.size(-2)  # = max_dist

            # [B,N,N,max_dist,H] -> [max_dist, B*N*N, H]
            hop_flat = hop_emb.permute(3, 0, 1, 2, 4).reshape(
                max_dist, -1, self.num_heads
            )

            # distance mixing matrices: reshape embedding weights
            # edge_dis_encoder.weight: [num_edge_dis * H * H, 1]
            mix_mats = self.edge_dis_encoder.weight.view(
                -1, self.num_heads, self.num_heads
            )  # [num_edge_dis, H, H]
            mix_mats = mix_mats[:max_dist]  # [max_dist, H, H]

            # batch matmul over "batch" = max_dist
            hop_mixed = torch.bmm(hop_flat, mix_mats)  # [max_dist, B*N*N, H]

            # reshape back: [B,N,N,max_dist,H]
            hop_mixed = hop_mixed.reshape(
                max_dist, B, N, N, self.num_heads
            ).permute(1, 2, 3, 0, 4)

            # sum hops and normalize by SPD
            hop_bias = hop_mixed.sum(-2) / (spatial_pos_.float().unsqueeze(-1))
            hop_bias = hop_bias.permute(0, 3, 1, 2)  # [B,H,N,N]

            graph_attn_bias[:, :, 1:, 1:] += hop_bias
        else:
            # baseline, non-multi-hop
            edge_bias = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)
            graph_attn_bias[:, :, 1:, 1:] += edge_bias

        # Re add base bias, as in original implementation
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # [B,H,N+1,N+1]

        # === Multi-channel gating on heads (optional) ===
        if self.use_multi_channel and self.num_local_heads is not None and self.num_local_heads > 0:
            # Build SPD_full over tokens [0..N] where index 0 is graph token
            T = N + 1
            spd_full = torch.zeros(
                B, T, T,
                dtype=spatial_pos.dtype,
                device=spatial_pos.device,
            )
            # Copy node-node SPD
            spd_full[:, 1:, 1:] = spatial_pos
            # Set distance 1 between graph token and all nodes
            spd_full[:, 0, 1:] = 1
            spd_full[:, 1:, 0] = 1
            spd_full[:, 0, 0] = 0

            # Local heads should not attend beyond local_radius
            local_mask = spd_full > self.local_radius  # [B,T,T] bool
            # Expand over the first num_local_heads
            local_mask = local_mask.unsqueeze(1).expand(B, self.num_local_heads, T, T)

            # Apply large negative bias to disallowed positions
            graph_attn_bias[:, :self.num_local_heads].masked_fill_(local_mask, float("-inf"))

        return graph_attn_bias


class MultiheadAttention(nn.Module):
    """Self attention with per head bias. Inputs [T, B, C]."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        query: torch.Tensor,                      # [T, B, C]
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],        # [B, H, T, T]
        key_padding_mask: Optional[torch.Tensor] = None,   # [B, T]
        attn_mask: Optional[torch.Tensor] = None,          # [T, T]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        T, B, C = query.size()
        assert C == self.embed_dim
        H = self.num_heads

        k_src = query if key is None else key
        v_src = query if value is None else value

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(k_src)
        v = self.v_proj(v_src)

        q = q.view(T, B * H, self.head_dim).transpose(0, 1)  # [B*H, T, d]
        k = k.view(T, B * H, self.head_dim).transpose(0, 1)
        v = v.view(T, B * H, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))       # [B*H, T, T]

        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias.view(B * H, T, T)

        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask.view(1, T, T)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            mask = mask.expand(B, H, T, T).reshape(B * H, T, T)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        attn = torch.bmm(attn_probs, v)                       # [B*H, T, d]
        attn = attn.transpose(0, 1).contiguous().view(T, B, C)
        attn = self.out_proj(attn)

        return attn, attn_probs


class GraphormerLayer(nn.Module):
    """Encoder block: self attention + FFN."""

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.self_attn = MultiheadAttention(
            embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout_module = nn.Dropout(dropout)
        act = nn.GELU()

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ffn_embedding_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(ffn_embedding_dim, embedding_dim),
        )
        self.final_layer_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        x: torch.Tensor,                      # [T, B, C]
        self_attn_bias: Optional[torch.Tensor] = None,  # [B, H, T, T]
        self_attn_mask: Optional[torch.Tensor] = None,  # [T, T]
        self_attn_padding_mask: Optional[torch.Tensor] = None,  # [B, T]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self attention
        residual = x
        x = self.self_attn_layer_norm(x)

        x_attn, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_bias=self_attn_bias,
            key_padding_mask=self_attn_padding_mask,
            attn_mask=self_attn_mask,
        )
        x = residual + self.dropout_module(x_attn)

        # FFN
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = residual + self.dropout_module(x)

        return x, attn


class GraphormerGraphEncoder(nn.Module):
    """
    Graphormer encoder that outputs:
      inner states (list of [T, B, C])
      graph representation [B, C] from graph token at index 0
    """

    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            n_layers=num_encoder_layers,
            use_multi_channel=True
        )

        self.emb_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.input_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.graph_norm = nn.LayerNorm(self.embedding_dim)

        self.layers = nn.ModuleList(
            [
                GraphormerLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout_module = nn.Dropout(dropout)
        self.apply(init_graphormer_params)

    def forward(
        self,
        batched: Batch,
        last_state_only: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        extra_node_features: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            batched["x"]: [B, N, 1]
        Returns:
            inner_states: list of [T, B, C]
            graph_rep: [B, C] from graph token
        """
        x_in = batched["x"]
        B, N = x_in.size()[:2]

        # Padding mask for nodes including graph token at position 0
        padding_mask = x_in[:, :, 0].eq(0)  # [B, N]
        padding_mask_cls = torch.zeros(B, 1, dtype=padding_mask.dtype, device=padding_mask.device)
        padding_mask = torch.cat([padding_mask_cls, padding_mask], dim=1)  # [B, N+1]

        # Node features with graph token
        x = self.graph_node_feature(batched)  # [B, N+1, C]
        if extra_node_features is not None:
            x[:, 1:, :] = x[:, 1:, :] + extra_node_features
        x = self.input_layer_norm(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout_module(x)

        attn_bias_full = self.graph_attn_bias(batched)  # [B, H, T, T]

        # B, T, C -> T, B, C
        x = x.transpose(0, 1)  # [T, B, C]
        T = x.size(0)

        if attn_mask is not None:
            assert attn_mask.shape == (T, T)

        inner_states: List[torch.Tensor] = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_bias=attn_bias_full,
                self_attn_mask=attn_mask,
                self_attn_padding_mask=padding_mask,
            )
            if not last_state_only:
                inner_states.append(x)

        if last_state_only:
            inner_states = [x]

        graph_rep = self.graph_norm(x[0, :, :])
        return inner_states, graph_rep


class GlycanGraphormerEncoder(nn.Module):
    """Wrapper around GraphormerGraphEncoder with glycan batching and caching."""

    def __init__(
        self,
        num_node_types: int = GLYCO_VOCAB_SIZE,
        num_edge_types: int = GLYCO_VOCAB_SIZE,
        max_in_degree: int = 32,
        max_out_degree: int = 32,
        max_spatial_distance: int = 32,
        multi_hop_max_dist: int = 5,
        num_layers: int = 6,
        hidden_dim: int = 256,
        ffn_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        mono_to_smiles: Optional[Dict[str, str]] = None,
        atom_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.max_spatial_distance = max_spatial_distance
        self.multi_hop_max_dist = multi_hop_max_dist
        self.num_heads = num_heads
        self.atom_hidden_dim = atom_hidden_dim or hidden_dim
        self.mono_to_smiles = mono_to_smiles or {}
        self.max_atomic_num = 118
        self.atom_mono_type = self.max_atomic_num + 1
        self.num_atom_types = self.atom_mono_type + 1
        self.atom_bond_types = {
            Chem.rdchem.BondType.SINGLE: 1,
            Chem.rdchem.BondType.DOUBLE: 2,
            Chem.rdchem.BondType.TRIPLE: 3,
            Chem.rdchem.BondType.AROMATIC: 4,
        }
        self.atom_special_edge = len(self.atom_bond_types) + 1
        self.num_atom_edge_types = self.atom_special_edge
        self.atom_max_in_degree = 8
        self.atom_max_out_degree = 8
        self.atom_max_spatial_distance = 8
        self.atom_multi_hop_max_dist = 3

        self.encoder = GraphormerGraphEncoder(
            num_atoms=num_node_types,
            num_in_degree=max_in_degree,
            num_out_degree=max_out_degree,
            num_edges=num_edge_types,
            num_spatial=max_spatial_distance + 1,
            num_edge_dis=multi_hop_max_dist + 1,
            edge_type="multi_hop",
            multi_hop_max_dist=multi_hop_max_dist,
            num_encoder_layers=num_layers,
            embedding_dim=hidden_dim,
            ffn_embedding_dim=ffn_dim,
            num_attention_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout
        )

        self.atom_encoder = GraphormerGraphEncoder(
            num_atoms=self.num_atom_types,
            num_in_degree=self.atom_max_in_degree,
            num_out_degree=self.atom_max_out_degree,
            num_edges=self.num_atom_edge_types,
            num_spatial=self.atom_max_spatial_distance + 1,
            num_edge_dis=self.atom_multi_hop_max_dist + 1,
            edge_type="multi_hop",
            multi_hop_max_dist=self.atom_multi_hop_max_dist,
            num_encoder_layers=max(2, num_layers // 2),
            embedding_dim=self.atom_hidden_dim,
            ffn_embedding_dim=self.atom_hidden_dim,
            num_attention_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.atom_to_mono = nn.Linear(self.atom_hidden_dim, hidden_dim) if self.atom_hidden_dim != hidden_dim else None
        self.atom_norm = nn.LayerNorm(hidden_dim)
        self.atom_dropout = nn.Dropout(dropout)

        self.cache: Dict[str, GlycanGraph] = {}
        self.tensor_cache: Dict[str, torch.Tensor] = {}
        self.atom_cache: Dict[str, GlycanGraph] = {}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ---------- graph and batch construction ----------

    def get_graph(self, iupac: str) -> GlycanGraph:
        g = self.cache.get(iupac)
        if g is None:
            g = iupac_to_glycan_graph(iupac)
            self.cache[iupac] = g
        return g

    def graphs_from_iupacs(self, iupacs: List[str]) -> List[GlycanGraph]:
        return [self.get_graph(i) for i in iupacs]

    def mono_name(self, mono_id: int) -> Optional[str]:
        if mono_id <= 0 or mono_id > len(GLYCOWORDS):
            return None
        return GLYCOWORDS[mono_id - 1]

    def smiles_graph(self, smiles: str) -> GlycanGraph:
        cached = self.atom_cache.get(smiles)
        if cached is not None:
            return cached
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            g = GlycanGraph(node_types=[], edge_index=[], edge_types=[])
            self.atom_cache[smiles] = g
            return g

        node_types: List[int] = []
        edge_index: List[Tuple[int, int]] = []
        edge_types: List[int] = []

        for atom in mol.GetAtoms():
            t = min(atom.GetAtomicNum(), self.max_atomic_num) + 1
            node_types.append(t)

        mono_idx = len(node_types)
        node_types.append(self.atom_mono_type)

        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bond_type = self.atom_bond_types.get(bond.GetBondType(), 1)
            edge_index.append((u, v))
            edge_types.append(bond_type)

        for atom_idx in range(mono_idx):
            edge_index.append((atom_idx, mono_idx))
            edge_types.append(self.atom_special_edge)

        g = GlycanGraph(node_types=node_types, edge_index=edge_index, edge_types=edge_types)
        self.atom_cache[smiles] = g
        return g

    def build_batch(self, graphs: List[GlycanGraph]) -> Batch:
        """Convert list of GlycanGraph objects into padded tensors."""
        B = len(graphs)
        if B == 0:
            raise ValueError("build_batch called with empty graph list")

        max_nodes = max(len(g.node_types) for g in graphs) or 1
        N = max_nodes
        device = self.device
        H = self.encoder.layers[0].self_attn.num_heads

        x = torch.zeros(B, N, 1, dtype=torch.long, device=device)
        in_degree = torch.zeros(B, N, dtype=torch.long, device=device)
        out_degree = torch.zeros(B, N, dtype=torch.long, device=device)
        spatial_pos = torch.zeros(B, N, N, dtype=torch.long, device=device)
        attn_bias = torch.zeros(B, N + 1, N + 1, dtype=torch.float32, device=device)
        attn_edge_type = torch.zeros(B, N, N, 1, dtype=torch.long, device=device)
        edge_input = torch.zeros(
            B,
            N,
            N,
            self.multi_hop_max_dist,
            H,
            dtype=torch.long,
            device=device,
        )

        for b, g in enumerate(graphs):
            num_nodes = len(g.node_types)
            if num_nodes == 0:
                continue

            self.precompute_graphormer_cache(g)

            x[b, :num_nodes, 0] = torch.tensor(g.node_types, dtype=torch.long, device=device)
            in_degree[b, :num_nodes] = g.in_degree.to(device)
            out_degree[b, :num_nodes] = g.out_degree.to(device)
            spatial_pos[b, :num_nodes, :num_nodes] = g.spatial_pos.to(device)
            attn_edge_type[b, :num_nodes, :num_nodes, 0] = g.attn_edge_type.to(device)

            ei = g.edge_input.to(device)  # [Ng, Ng, max_dist]
            ei_expanded = ei.unsqueeze(-1).expand(-1, -1, -1, H)
            edge_input[b, :num_nodes, :num_nodes, :, :] = ei_expanded

        in_degree = in_degree.clamp(max=self.max_in_degree - 1)
        out_degree = out_degree.clamp(max=self.max_out_degree - 1)

        return {
            "x": x,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "attn_bias": attn_bias,
            "spatial_pos": spatial_pos,
            "attn_edge_type": attn_edge_type,
            "edge_input": edge_input,
        }

    def precompute_atom_graphormer_cache(self, g: GlycanGraph) -> None:
        if (
            getattr(g, "spatial_pos", None) is not None
            and getattr(g, "edge_input", None) is not None
            and getattr(g, "max_dist_cached", None) == self.atom_multi_hop_max_dist
            and getattr(g, "max_spatial_cached", None) == self.atom_max_spatial_distance
        ):
            return

        num_nodes = len(g.node_types)
        if num_nodes == 0:
            g.in_degree = torch.zeros(0, dtype=torch.long)
            g.out_degree = torch.zeros(0, dtype=torch.long)
            g.spatial_pos = torch.zeros(0, 0, dtype=torch.long)
            g.attn_edge_type = torch.zeros(0, 0, dtype=torch.long)
            g.edge_input = torch.zeros(0, 0, self.atom_multi_hop_max_dist, dtype=torch.long)
            g.max_dist_cached = self.atom_multi_hop_max_dist
            g.max_spatial_cached = self.atom_max_spatial_distance
            return

        device_cpu = torch.device("cpu")

        in_degree = torch.zeros(num_nodes, dtype=torch.long, device=device_cpu)
        out_degree = torch.zeros(num_nodes, dtype=torch.long, device=device_cpu)
        spatial_pos = torch.zeros(num_nodes, num_nodes, dtype=torch.long, device=device_cpu)
        attn_edge_type = torch.zeros(num_nodes, num_nodes, dtype=torch.long, device=device_cpu)
        edge_input = torch.zeros(
            num_nodes,
            num_nodes,
            self.atom_multi_hop_max_dist,
            dtype=torch.long,
            device=device_cpu,
        )

        adj: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
        for (u, v), e_type in zip(g.edge_index, g.edge_types):
            u = int(u)
            v = int(v)
            e_type = int(e_type)

            adj[u].append((v, e_type))
            adj[v].append((u, e_type))

            in_degree[u] += 1
            in_degree[v] += 1
            out_degree[u] += 1
            out_degree[v] += 1

            attn_edge_type[u, v] = e_type
            attn_edge_type[v, u] = e_type

        for src in range(num_nodes):
            dist = [-1] * num_nodes
            prev_node = [-1] * num_nodes
            prev_edge = [0] * num_nodes

            dist[src] = 0
            queue = [src]
            head = 0

            while head < len(queue):
                u = queue[head]
                head += 1
                du = dist[u]

                if du >= self.atom_multi_hop_max_dist:
                    continue

                for v, e_type in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        prev_node[v] = u
                        prev_edge[v] = e_type
                        queue.append(v)

            for dst in range(num_nodes):
                d = dist[dst]
                if d < 0:
                    continue
                d_clamped = min(d, self.atom_max_spatial_distance)
                spatial_pos[src, dst] = d_clamped

                if d <= 0:
                    continue

                path: List[int] = []
                cur = dst
                steps = 0
                while cur != src and cur != -1 and steps < self.atom_multi_hop_max_dist:
                    e_type = prev_edge[cur]
                    if e_type == 0:
                        break
                    path.append(e_type)
                    cur = prev_node[cur]
                    steps += 1

                path = path[::-1]
                for hop_idx, e_type in enumerate(path):
                    if hop_idx >= self.atom_multi_hop_max_dist:
                        break
                    edge_input[src, dst, hop_idx] = e_type

        g.in_degree = in_degree
        g.out_degree = out_degree
        g.spatial_pos = spatial_pos
        g.attn_edge_type = attn_edge_type
        g.edge_input = edge_input
        g.max_dist_cached = self.atom_multi_hop_max_dist
        g.max_spatial_cached = self.atom_max_spatial_distance

    def build_atom_batch(self, graphs: List[GlycanGraph]) -> Batch:
        B = len(graphs)
        if B == 0:
            raise ValueError("build_atom_batch called with empty graph list")

        max_nodes = max(len(g.node_types) for g in graphs) or 1
        N = max_nodes
        device = self.device
        H = self.atom_encoder.layers[0].self_attn.num_heads

        x = torch.zeros(B, N, 1, dtype=torch.long, device=device)
        in_degree = torch.zeros(B, N, dtype=torch.long, device=device)
        out_degree = torch.zeros(B, N, dtype=torch.long, device=device)
        spatial_pos = torch.zeros(B, N, N, dtype=torch.long, device=device)
        attn_bias = torch.zeros(B, N + 1, N + 1, dtype=torch.float32, device=device)
        attn_edge_type = torch.zeros(B, N, N, 1, dtype=torch.long, device=device)
        edge_input = torch.zeros(
            B,
            N,
            N,
            self.atom_multi_hop_max_dist,
            H,
            dtype=torch.long,
            device=device,
        )

        clamp_in = self.atom_max_in_degree - 1
        clamp_out = self.atom_max_out_degree - 1

        for b, g in enumerate(graphs):
            num_nodes = len(g.node_types)
            if num_nodes == 0:
                continue

            self.precompute_atom_graphormer_cache(g)

            x[b, :num_nodes, 0] = torch.tensor(g.node_types, dtype=torch.long, device=device)

            if g.in_degree is not None:
                in_degree[b, :num_nodes] = g.in_degree.to(device).clamp(max=clamp_in)
            if g.out_degree is not None:
                out_degree[b, :num_nodes] = g.out_degree.to(device).clamp(max=clamp_out)
            if g.spatial_pos is not None:
                spatial_pos[b, :num_nodes, :num_nodes] = g.spatial_pos.to(device)
            if g.attn_edge_type is not None:
                attn_edge_type[b, :num_nodes, :num_nodes, 0] = g.attn_edge_type.to(device)
            if g.edge_input is not None:
                edge_input[b, :num_nodes, :num_nodes, :, :] = g.edge_input.to(device).unsqueeze(-1).repeat(1, 1, 1, 1, H)

        attn_mask = x[:, :, 0].eq(0)
        attn_bias[:, 1:, 1:][attn_mask] = float("-inf")
        attn_bias[:, 1:, 1:][attn_mask.unsqueeze(2)] = float("-inf")

        return {
            "x": x,
            "in_degree": in_degree,
            "out_degree": out_degree,
            "attn_bias": attn_bias,
            "spatial_pos": spatial_pos,
            "attn_edge_type": attn_edge_type,
            "edge_input": edge_input,
        }

    def build_atom_features(self, graphs: List[GlycanGraph], batched: Batch) -> torch.Tensor:
        B, N = batched["x"].shape[:2]
        device = self.device
        target_dim = self.encoder.embedding_dim
        atom_features = torch.zeros(B, N, target_dim, device=device)

        atom_graphs: List[GlycanGraph] = []
        owners: List[Tuple[int, int]] = []
        for b, g in enumerate(graphs):
            for mono_idx, mono_id in enumerate(g.node_types):
                if mono_id == 0:
                    continue
                mono_name = self.mono_name(mono_id)
                smiles = None if mono_name is None else self.mono_to_smiles.get(mono_name)
                if not smiles:
                    continue
                atom_graphs.append(self.smiles_graph(smiles))
                owners.append((b, mono_idx))

        if not atom_graphs:
            return atom_features

        atom_batch = self.build_atom_batch(atom_graphs)
        atom_inner, _ = self.atom_encoder(atom_batch, last_state_only=True)
        atom_last = atom_inner[-1][1:, :, :].transpose(0, 1)
        atom_mask = ~atom_batch["x"][:, :, 0].eq(0)

        for i, (b, m) in enumerate(owners):
            mask = atom_mask[i]
            if not mask.any():
                continue
            pooled = atom_last[i][mask].mean(dim=0)
            if self.atom_to_mono is not None:
                pooled = self.atom_to_mono(pooled)
            atom_features[b, m] = pooled

        atom_features = self.atom_dropout(atom_features)
        atom_features = self.atom_norm(atom_features)
        return atom_features

    def precompute_graphormer_cache(self, g: GlycanGraph) -> None:
        """Cache SPD, degrees, edge types and multi hop paths on CPU for a single graph."""
        if (
            getattr(g, "spatial_pos", None) is not None
            and getattr(g, "edge_input", None) is not None
            and getattr(g, "max_dist_cached", None) == self.multi_hop_max_dist
            and getattr(g, "max_spatial_cached", None) == self.max_spatial_distance
        ):
            return

        num_nodes = len(g.node_types)
        if num_nodes == 0:
            g.in_degree = torch.zeros(0, dtype=torch.long)
            g.out_degree = torch.zeros(0, dtype=torch.long)
            g.spatial_pos = torch.zeros(0, 0, dtype=torch.long)
            g.attn_edge_type = torch.zeros(0, 0, dtype=torch.long)
            g.edge_input = torch.zeros(0, 0, self.multi_hop_max_dist, dtype=torch.long)
            g.max_dist_cached = self.multi_hop_max_dist
            g.max_spatial_cached = self.max_spatial_distance
            return

        device_cpu = torch.device("cpu")

        in_degree = torch.zeros(num_nodes, dtype=torch.long, device=device_cpu)
        out_degree = torch.zeros(num_nodes, dtype=torch.long, device=device_cpu)
        spatial_pos = torch.zeros(num_nodes, num_nodes, dtype=torch.long, device=device_cpu)
        attn_edge_type = torch.zeros(num_nodes, num_nodes, dtype=torch.long, device=device_cpu)
        edge_input = torch.zeros(
            num_nodes,
            num_nodes,
            self.multi_hop_max_dist,
            dtype=torch.long,
            device=device_cpu,
        )

        # adjacency with edge types
        adj: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
        for (u, v), e_type in zip(g.edge_index, g.edge_types):
            u = int(u)
            v = int(v)
            e_type = int(e_type)

            adj[u].append((v, e_type))
            adj[v].append((u, e_type))

            in_degree[u] += 1
            in_degree[v] += 1
            out_degree[u] += 1
            out_degree[v] += 1

            attn_edge_type[u, v] = e_type
            attn_edge_type[v, u] = e_type

        # BFS per source
        for src in range(num_nodes):
            dist = [-1] * num_nodes
            prev_node = [-1] * num_nodes
            prev_edge = [0] * num_nodes  # edge_type from parent

            dist[src] = 0
            queue = [src]
            head = 0

            while head < len(queue):
                u = queue[head]
                head += 1
                du = dist[u]

                if du >= self.multi_hop_max_dist:
                    continue

                for v, e_type in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        prev_node[v] = u
                        prev_edge[v] = e_type
                        queue.append(v)

            for dst in range(num_nodes):
                d = dist[dst]
                if d < 0:
                    continue
                d_clamped = min(d, self.max_spatial_distance)
                spatial_pos[src, dst] = d_clamped

                if d <= 0:
                    continue

                # Reconstruct path edges src -> dst
                path: List[int] = []
                cur = dst
                steps = 0
                while cur != src and cur != -1 and steps < self.multi_hop_max_dist:
                    e_type = prev_edge[cur]
                    if e_type == 0:
                        break
                    path.append(e_type)
                    cur = prev_node[cur]
                    steps += 1

                path = path[::-1]
                for hop_idx, e_type in enumerate(path):
                    if hop_idx >= self.multi_hop_max_dist:
                        break
                    edge_input[src, dst, hop_idx] = e_type

        g.in_degree = in_degree
        g.out_degree = out_degree
        g.spatial_pos = spatial_pos
        g.attn_edge_type = attn_edge_type
        g.edge_input = edge_input
        g.max_dist_cached = self.multi_hop_max_dist
        g.max_spatial_cached = self.max_spatial_distance

    # ---------- high level encoding ----------

    def forward(
        self,
        iupacs: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            iupacs: list of IUPAC strings, len = B
        Returns:
            g_pad: [B, N, H] node embeddings (padded)
            g_mask: [B, N] bool mask for real nodes
            graph_rep: [B, H] graph token embeddings
        """
        graphs = self.graphs_from_iupacs(iupacs)
        batched = self.build_batch(graphs)
        atom_features = self.build_atom_features(graphs, batched)

        inner_states, graph_rep = self.encoder(
            batched,
            last_state_only=True,
            extra_node_features=atom_features,
        )
        last_state = inner_states[-1]  # [T, B, H]
        node_states = last_state[1:, :, :].transpose(0, 1)  # [B, N, H]

        data_x = batched["x"]
        padding_mask = data_x[:, :, 0].eq(0)  # [B, N]
        g_mask = ~padding_mask

        return node_states, g_mask, graph_rep


class GlycanGraphormerPretrainer(nn.Module):
    """
    Masked node pretraining for GlycanGraphormerEncoder.
    Predict node type ids from final node states.
    """

    def __init__(
        self,
        glycan_encoder: GlycanGraphormerEncoder,
        vocab_size: int = GLYCO_VOCAB_SIZE,
        mask_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.encoder = glycan_encoder
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob

        hidden_dim = self.encoder.encoder.embedding_dim
        self.node_head = nn.Linear(hidden_dim, vocab_size)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    @property
    def device(self) -> torch.device:
        return self.encoder.device

    # ---------- masking utilities ----------

    def build_batched_graphs(self, iupacs: List[str]) -> Batch:
        graphs = self.encoder.graphs_from_iupacs(iupacs)
        return self.encoder.build_batch(graphs)

    def apply_node_masking(
        self,
        batched: Batch,
    ) -> Tuple[Batch, torch.Tensor]:
        """
        Mask node ids in batched["x"] and build labels.
        Returns:
            batched_masked: same dict with masked x
            labels: [B, N] with -100 where loss is ignored
        """
        x = batched["x"]  # [B, N, 1]
        assert x.dim() == 3 and x.size(2) == 1

        orig_ids = x[:, :, 0]  # [B, N]
        labels = orig_ids.clone()

        is_valid = orig_ids.ne(0)  # id 0 is pad

        rand = torch.rand_like(orig_ids.float(), device=orig_ids.device)
        mask = is_valid & (rand < self.mask_prob)

        labels[~mask] = -100  # ignore_index

        masked_ids = orig_ids.clone()
        masked_ids[mask] = 0  # reuse pad id as mask id

        batched_masked = dict(batched)
        batched_masked["x"] = masked_ids.unsqueeze(-1)

        return batched_masked, labels

    # ---------- training and encoding ----------

    def forward(
        self,
        iupacs: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            iupacs: list of IUPAC condensed glycan strings
        Returns:
            dict with loss, logits, labels, mask, graph_rep
        """
        device = self.device

        graphs = self.encoder.graphs_from_iupacs(iupacs)
        batched = self.encoder.build_batch(graphs)
        atom_features = self.encoder.build_atom_features(graphs, batched)
        batched = {k: v.to(device) for k, v in batched.items()}

        batched_masked, labels = self.apply_node_masking(batched)
        mask = labels.ne(-100)

        inner_states, graph_rep = self.encoder.encoder(
            batched_masked,
            last_state_only=True,
            attn_mask=None,
            extra_node_features=atom_features,
        )
        last_state = inner_states[-1]  # [T, B, H]
        node_states = last_state[1:, :, :].transpose(0, 1)  # [B, N, H]

        logits = self.node_head(node_states)  # [B, N, V]

        loss = self.loss_fct(
            logits.view(-1, self.vocab_size),
            labels.view(-1),
        )

        return {
            "loss": loss,
            "logits": logits,
            "labels": labels,
            "mask": mask,
            "graph_rep": graph_rep,
        }

    @torch.no_grad()
    def encode(
        self,
        iupacs: List[str],
    ) -> torch.Tensor:
        """Convenience method to get graph level embeddings [B, H]."""
        self.eval()
        graphs = self.encoder.graphs_from_iupacs(iupacs)
        batched = self.encoder.build_batch(graphs)
        atom_features = self.encoder.build_atom_features(graphs, batched)
        batched = {k: v.to(self.device) for k, v in batched.items()}
        _, graph_rep = self.encoder.encoder(
            batched,
            last_state_only=True,
            attn_mask=None,
            extra_node_features=atom_features,
        )
        return graph_rep
