import pandas as pd
from lightgbm import LGBMModel
import numpy as np
import lightgbm as lgbm

FOLDER_NAME = "data/ml_model_data"


def column_name_df_preds(model_name: str) -> str:
    return "pred_" + model_name.split("_")[-1].split(".bin")[0]


def save_pred_df(df: pd.DataFrame, data_set_string: str) -> None:
    df.to_csv(
        f"{FOLDER_NAME}/preds/preds_load_dump_{data_set_string}.csv",
        sep=",",
        index=False,
    )


def plot_metrics(
    booster: dict | LGBMModel,
    metric: str | None,
    ax: np.ndarray,
    load_or_dump: str,
) -> None:
    lgbm.plot_metric(
        booster=booster,
        metric=metric,
        ax=ax,
        grid=True,
        title=f"Learning curve {load_or_dump}",
        ylabel="MSE (Mean Squared Error)",
    )
