import pathlib
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import arviz as az
from theano import tensor as tt
from . import preprocess

coarse_emo_list = preprocess.coarse_emo_list
emotion_labels = coarse_emo_list


def bayesian_model(df: pd.DataFrame, drop_ot: bool =False, dfac: float=1.5) -> pm.Model:
  df = df.copy()
  if drop_ot:
    df = df[df["dc"] != "OT"].reset_index(drop=True)

  sz_idx, sz_names = pd.factorize(df["dc"], sort=True)
  stim_idx, stim_names = pd.factorize(df["stim"], sort=True)
  pid_idx, pid_names = pd.factorize(df["pid"], sort=True, )
  with pm.Model(
      coords={
          "pid": pid_names,
          "sz": sz_names,
          "stim": stim_names,
          "emo": emotion_labels,
          "emo_to": emotion_labels,
          "trial": np.arange(len(df)),
      }
  ) as model:
    emotion_go_vals = pm.Data("emotion_go_vals", value=df[emotion_labels], dims=["trial", "emo"],)
    # Normative vectors
    normative_emo = pm.Dirichlet(
          "normative_emo",
          a=np.ones(len(emotion_labels), dtype=np.float32) / (dfac * len(emotion_labels)),
          dims=["stim", "emo"],
      )
      # SZ Rotation
    beta = pm.Dirichlet(
          "beta",
          a=np.ones(len(emotion_labels), dtype=np.float32) / (dfac * len(emotion_labels)),
          dims=["emo", "emo_to"],
      )

    dummy_mapper = pm.Data(
          "dummy_mapper", np.eye(len(emotion_labels)), dims=["emo", "emo_to"]
      )
    # Random effects
    # Prior on the diagonal parameters of the random rotation effect 
    # dirichlet dist
    re_diag_mu  = 10.0
    re_diag = pm.Gamma("re_diag", mu=re_diag_mu, sigma=re_diag_mu / np.sqrt(2))

    # Prior on the off-center parameters of the random rotation effect 
    # dirichlet dist
    re_rest_mu = 1.0
    re_rest = pm.Gamma("re_rest", mu=re_rest_mu, sigma = re_rest_mu / np.sqrt(2))
    # Random effects. Tthe diagonals are really re_diag + re_rest, which
    # ensures that they're larger than the off diagonals. Sort of equivalent
    # to mean 0 prior on linear random effects?
    random_effects = pm.Dirichlet(
          "random_effects",
          a=(
              re_rest
              * np.ones((len(emotion_labels), len(emotion_labels)), dtype=np.float32)
          )
          + (re_diag * np.eye(len(emotion_labels), dtype=np.float32)),
          dims=["pid", "emo", "emo_to"],
      )
    # Expand the random effects to each trial
    re_vals = random_effects[..., pid_idx, :, :]  # dims=["trial", "emo", "emo_to"]

    # To get the group rotated emotions take the base emotions with dims
    # [stim, emo, 1], multiply that by (beta - I) with dims [1, emo, emo_to],
    # then sum over the emo dimension so the result is [stim, emo_to]
    normative_emo_group_perturbation = tt.sum(
          tt.mul(normative_emo[..., None], beta), -2
      )  # dims=[stim, emo_to]

    # Expand normative and rotated emotions over trials
    normative_emo_trial = normative_emo[..., stim_idx, :]  # dims=["trial", "emo"]
    normative_emo_group_perturbation_trial = normative_emo_group_perturbation[
        ..., stim_idx, :
    ]  # dims=["trial", "emo"]

    sz_bool = pm.Data("sz_bool", (df["dc"] == "PL").astype(float), dims=["trial"])

    group_perturbation = sz_bool[..., :, None] * normative_emo_group_perturbation_trial
    mu = ((1.0 - sz_bool[..., :, None]) * normative_emo_trial) + (sz_bool[..., :, None] * group_perturbation)  # dims=["trial", "emo"]


    if not drop_ot:

      beta_drug = pm.Dirichlet(
        "beta_drug",
        a=np.ones(len(emotion_labels), dtype=np.float32) / (dfac * len(emotion_labels)),
        dims=["emo", "emo_to"],
      )
      normative_emo_drug_perturbation_trial = tt.sum(
        tt.mul(mu[..., None], beta_drug), -2
    )  # dims=[trial, emo_to]

      ot_bool = pm.Data("ot_bool", (df["dc"] == "OT").astype(float), dims=["trial"])

      mu = ((1.0 - ot_bool[..., :, None]) * mu) + (ot_bool[..., :, None] * normative_emo_drug_perturbation_trial)

    mu_re = tt.sum(
        tt.mul(mu[..., :, :, None], re_vals), -2
    )  # dims=["trial", "emo"]

    obs_mag = pm.HalfCauchy("obs_mag", 0.5)

    pm.Dirichlet("p", 
                 a=mu_re * obs_mag, 
                 observed=emotion_go_vals, 
                 dims=["trial", "emo"],)
  return model

if __name__=="__main__":
    data_path = pathlib.Path("go_emotions_output_clean.csv")
    with data_path.open("r") as fl:
        raw_data = pd.read_csv(fl)
    df = preprocess.fine_to_coarse(raw_data)

    eps = 1e-6
    for emo in coarse_emo_list:
        df.loc[df[emo] < eps, emo] = eps
        df.loc[df[emo] > 1.0 - eps, emo] = 1.0 - eps

    df[coarse_emo_list] = df[coarse_emo_list].divide(
        df[coarse_emo_list].sum(1), 0
    )

    drop_ot = False
    model = bayesian_model(df, drop_ot, 1.0)
    temp_trace = pm.sample(
        1500, model=model, return_inferencedata=True, tune=2500, chains=4
    )
    trace_vars = ["re_diag", "re_rest", "beta", "obs_mag", "normative_emo"]
    if not drop_ot:
        trace_vars.append("beta_drug")
    az.plot_trace(
        temp_trace,
        var_names=trace_vars,
        compact=True,
    )

    # display(temp_trace.posterior["beta"].mean(["chain", "draw"]).to_series().unstack())
    # display(temp_trace.posterior["beta_drug"].mean(["chain", "draw"]).to_series().unstack())