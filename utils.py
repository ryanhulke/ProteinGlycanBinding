import typing as T
import logging as lg
import sys
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

logLevels = {0: lg.ERROR, 1: lg.WARNING, 2: lg.INFO, 3: lg.DEBUG}
LOGGER_NAME = "DTI"


def get_logger(logger_name: str = None) -> lg.Logger:
    if logger_name is None:
        logger_name = LOGGER_NAME
    return lg.getLogger(logger_name)


logg = get_logger()


def config_logger(
    file: T.Union[Path, None],
    fmt: str,
    level: bool = 2,
    use_stdout: bool = True,
):
    """
    Create and configure the logger

    :param file: Can be a Path or None -- if a Path, log messages will be written to the file at Path
    :type file: T.Union[Path, None]
    :param fmt: Formatting string for the log messages
    :type fmt: str
    :param level: Level of verbosity
    :type level: int
    :param use_stdout: Whether to also log messages to stdout
    :type use_stdout: bool
    :return:
    """

    module_logger = lg.getLogger(LOGGER_NAME)
    module_logger.setLevel(logLevels[level])
    formatter = lg.Formatter(fmt)

    if file is not None:
        fh = lg.FileHandler(file)
        fh.setFormatter(formatter)
        module_logger.addHandler(fh)

    if use_stdout:
        sh = lg.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        module_logger.addHandler(sh)

    lg.propagate = False

    return module_logger


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_pretrained_glycan_encoder(sweetbind_model, ckpt_path, device):
    print(f"[Loading pretrained glycan encoder from {ckpt_path}]")
    ckpt = torch.load(ckpt_path, map_location=device)

    full_state = ckpt["model_state_dict"]

    mapped = {}

    for k, v in full_state.items():

        if k.startswith("node_head"):
            continue

        # strip the double prefix
        if k.startswith("encoder.encoder."):
            k = k.replace("encoder.encoder.", "")

        # the target expects "encoder." at the front
        new_k = "encoder." + k

        mapped[new_k] = v

    # Actually load
    missing, unexpected = sweetbind_model.glycan_encoder.load_state_dict(
        mapped, strict=False
    )

    print("[Glycan encoder weights loaded into SweetBind]")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return sweetbind_model