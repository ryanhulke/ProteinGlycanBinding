from __future__ import annotations

import re
import pickle as pkl
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import torch
from glycowork.motif import tokenization
from glycowork.motif.graph import glycan_to_nxGraph

# Load vocab
VOCAB_PATH = "./data/glycoword_vocab.pkl"
ENTITIES: List[str] = pkl.load(open(VOCAB_PATH, "rb"))

# Entities contain both units and links. We also allow bracket tokens.
GLYCOWORDS: List[str] = ENTITIES + ["Unknown_Token"] #+ ["[", "]", "Unknown_Token", "[MASK]"]
GLYCOWORD2ID: Dict[str, int] = {w: i for i, w in enumerate(GLYCOWORDS)}
GLYCO_PAD_ID = 0
GLYCO_VOCAB_SIZE = len(GLYCOWORD2ID) + 1  # for graphormer. +1 for padding id
# Split units vs links to help tokenization
UNIT_VOCAB = [e for e in ENTITIES if not (e.startswith(("a", "b", "?")) or re.match(r"^[0-9]+(?:-[0-9]+)+$", e))] + ["<PAD>"]
LINK_VOCAB = [e for e in ENTITIES if (e.startswith(("a", "b", "?")) or re.match(r"^[0-9]+(?:-[0-9]+)+$", e))]
UNIT_VOCAB_SORTED = sorted(UNIT_VOCAB, key=len, reverse=True)

# Link inside parentheses like (a1-3), (b1-4), (?-3), or even (3-6)
LINK_PARENS_PATTERN = re.compile(r"\(([^\(\)]+)\)")

class TokenType:
    UNIT = "UNIT"
    LINK = "LINK"
    BRANCH_OPEN = "BRANCH_OPEN"
    BRANCH_CLOSE = "BRANCH_CLOSE"

@dataclass
class Token:
    type: str
    text: str
    start: int
    end: int

@dataclass
class GlycanGraph:
    node_types: List[int]
    edge_index: List[Tuple[int, int]]
    edge_types: List[int]
    # Cached Graphormer fields (CPU tensors)
    in_degree: Optional[torch.Tensor] = None          # [N]
    out_degree: Optional[torch.Tensor] = None         # [N]
    spatial_pos: Optional[torch.Tensor] = None        # [N, N]
    attn_edge_type: Optional[torch.Tensor] = None     # [N, N]
    edge_input: Optional[torch.Tensor] = None         # [N, N, max_dist]
    max_dist_cached: int = 0                          # multi_hop_max_dist used
    max_spatial_cached: int = 0                       # max_spatial_distance used

def get_node_label(node_data: Dict) -> str:
    candidates = [v for v in node_data.values() if isinstance(v, str)]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Cannot find node label in node_data keys={list(node_data.keys())}")


def get_edge_label(edge_data: Dict) -> Optional[str]:
    candidates = [v for v in edge_data.values() if isinstance(v, str)]
    if len(candidates) == 1:
        print("candidate: " + candidates[0])
        return candidates[0]
    return None # many edges returned by glycowork have no labels due to misformatted IUPAC

def iupac_to_glycan_graph(iupac: str) -> GlycanGraph:
    G = glycan_to_nxGraph(iupac)
    if G is None or len(G.nodes) == 0:
        print(f"Warning: could not parse IUPAC '{iupac}' into a graph.")
        return GlycanGraph(node_types=[], edge_index=[], edge_types=[])

    nodes = sorted(G.nodes())
    node_idx_map = {n: i for i, n in enumerate(nodes)}

    node_types: List[int] = []
    for n in nodes:
        data = G.nodes[n]
        raw_label = get_node_label(data)
        core = tokenization.get_core(raw_label)
        base_id = GLYCOWORD2ID.get(core, GLYCOWORD2ID["Unknown_Token"])
        node_id = base_id + 1  # shift by +1 so 0 is pad
        node_types.append(node_id)

    edge_index: List[Tuple[int, int]] = []
    edge_types: List[int] = []

    for u, v, data in G.edges(data=True):
        src = node_idx_map[u]
        dst = node_idx_map[v]

        edge_label = get_edge_label(data)
        if edge_label is None:
            base_edge_id = GLYCOWORD2ID["Unknown_Token"]
        else:
            base_edge_id = GLYCOWORD2ID.get(edge_label, GLYCOWORD2ID["Unknown_Token"])
        edge_type_id = base_edge_id + 1  # shift by +1; 0 reserved as pad

        # store each undirected edge once; we'll add both directions in build_batch
        edge_index.append((src, dst))
        edge_types.append(edge_type_id)

    return GlycanGraph(
        node_types=node_types,
        edge_index=edge_index,
        edge_types=edge_types,
    )

class IUPACTokenizer:
    def __init__(self,
                 unit_vocab: List[str] = None,
                 link_vocab: List[str] = None,
                 glycoword2id: Dict[str, int] = None):
        self.unit_vocab = unit_vocab or UNIT_VOCAB
        self.link_vocab = set(link_vocab or LINK_VOCAB)
        self.glycoword2id = glycoword2id or GLYCOWORD2ID
        # build a greedy matcher for units present in this tokenizer
        units_sorted = sorted(self.unit_vocab, key=len, reverse=True)
        self.unit_re = re.compile("|".join(re.escape(u) for u in units_sorted))

    def core(self, s: str) -> str:
        return tokenization.get_core(s)

    def read_until_delim(self, s: str, i: int) -> int:
        """Read a unit chunk until next '(', '[', ']'', or end."""
        n = len(s)
        while i < n and s[i] not in "()[\]":
            i += 1
        return i  # index of delimiter or n

    def chunk_to_unit(self, chunk: str) -> str:
        core = self.core(chunk)
        if core in self.unit_vocab:
            return core

        # fallback, find longest vocab substring inside chunk
        for cand in UNIT_VOCAB_SORTED:
            if cand in chunk:
                return cand
        raise ValueError(f"Cannot map chunk '{chunk}' to any known unit.")

    
    def tokenize(self, iupac: str) -> List[Token]:
        s = iupac.strip()
        s = re.sub(r"\s+", "", s)  # remove whitespace
        tokens: List[Token] = []
        # modifications: List[str] = []
        i = 0
        n = len(s)

        while i < n:
            c = s[i]
            if c == "[":
                i += 1
                continue
            if c == "]":
                i += 1
                continue

            # Parenthesized link like (a1-3) or (b1-4)
            if c == "(":
                m = LINK_PARENS_PATTERN.match(s, i)
                if not m:
                    break  # malformed or end link
                text = m.group(1)
                tokens.append(Token(TokenType.LINK, text, i, m.end()))
                i = m.end()
                continue

            # Otherwise we are at the start of a unit chunk, which may include modifiers and digits
            j = self.read_until_delim(s, i)
            if j == i:
                # safety, should not happen
                raise ValueError(f"Tokenizer stalled at {i} in {iupac}")
            chunk = s[i:j]
            unit = self.chunk_to_unit(chunk)
            tokens.append(Token(TokenType.UNIT, unit, i, j))
            i = j

        return tokens #, modifications

    def to_ids(self, tokens: List[Token]) -> List[int]:
        ids = []
        try:
            for t in tokens:
                key = t.text
                ids.append(self.glycoword2id.get(key, self.glycoword2id["Unknown_Token"]))
        except Exception as e:
            print(f"Error converting tokens to ids: {e}")
            ids = []
        return ids
