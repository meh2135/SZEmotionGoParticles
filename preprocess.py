"""Preprocessing utilities for SZEmotionsGO particle modeling."""
import itertools
import numpy as np
import pandas as pd
import yaml
import pathlib
from typing import Optional
from scipy.special import logit
from typing import List, Dict
default_coarse_to_fine_emo_map_path = pathlib.Path("./coarse_to_fine_emo_map.yaml")

def fix_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index(["id", "stimulus.x", "drugCondition"])
    df.index.names = ["pid", "stim", "dc"]
    return df


def normalize(df: pd.DataFrame, logit_first: bool =True, power: Optional[float] =None) -> pd.DataFrame:
    if logit_first and power is not None:
        raise ValueError("Taking the logit first will cause most values to be negative. Need to exp!")
    if logit_first:
        df = logit(df)
    if power is None:
        unscaled_df = np.exp(df)
    else:
        unscaled_df = df ** power
    return unscaled_df.divide(unscaled_df.sum(1), axis=0)


def coarsen(df: pd.DataFrame, coarse_to_fine_emo_map: Dict[str, List[str]]) -> pd.DataFrame:
    return pd.DataFrame({coarse_emo: df[fine_emo_list].sum(1) for coarse_emo, fine_emo_list in coarse_to_fine_emo_map.items()})

def bound(df: pd.DataFrame, delta: float) -> pd.DataFrame:
    df[df<delta] = delta
    df[df > (1.0 - delta)] = 1.0 - delta
    return df

def preprocess(df: pd.DataFrame, 
               logit_first: bool = True, 
               power: Optional[float] = None, 
               delta: float=1e-5, 
               emo_map_path: pathlib.Path = default_coarse_to_fine_emo_map_path
               ) -> pd.DataFrame:
    with emo_map_path.open("r") as fl:
        coarse_to_fine_emo_map = yaml.load(fl)
    clean_df = normalize(fix_column_names(df), logit_first, power)
    return bound(coarsen(clean_df, coarse_to_fine_emo_map), delta)