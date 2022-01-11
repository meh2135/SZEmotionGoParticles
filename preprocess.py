"""Preprocessing utilities for SZEmotionsGO particle modeling."""
import pandas as pd
import yaml
import pathlib
coarse_to_fine_emo_map_path = pathlib.Path("./coarse_to_fine_emo_map.yaml")
with coarse_to_fine_emo_map_path.open("r") as fl:
    coarse_to_fine_emo_map = yaml.load(fl)
coarse_emo_list = list(coarse_to_fine_emo_map.keys())

def fine_to_coarse(df: pd.DataFrame, 
                   agg_power: float = 2.0,
                   normalize: bool = True) -> pd.DataFrame:
    # Grab the non-emotion-value columns
    coarse_emo_df_dict = {"pid":df["id"], "stim":df["stim"], "dc":df["drugCondition"]}
    for coarse, fine_list in coarse_to_fine_emo_map.items():
        coarse_emo_df_dict[coarse] = (df[fine_list]**agg_power).sum(1)
    # Construct coarse emotion dataframe
    coarse_emo_df = pd.DataFrame(coarse_emo_df_dict)
    # Extract the rater assigned valence for each stimulus
    coarse_emo_df["stim_rating"] = coarse_emo_df["stim"].str.extract("([a-z]{3})_")
    # Normalize so that the sum of the 4 coarse emotions=1 for each trial.
    if normalize:
        coarse_emo_df[coarse_emo_list] = coarse_emo_df[coarse_emo_list].divide(coarse_emo_df[coarse_emo_list].sum(1), 0)
    # Clear the old index
    coarse_emo_df = coarse_emo_df.reset_index(drop=True)
    return coarse_emo_df