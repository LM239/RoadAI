import pandas as pd
from lightgbm import LGBMModel
import numpy as np
import lightgbm as lgbm
from pathlib import Path

FOLDER_NAME = "data/ml_model_data"
MODEL_FOLDER = Path(f"{FOLDER_NAME}/models")


class LightGBMParams:
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 31,
        "learning_rate": 0.5,
        "feature_fraction": 1,
        "num_boost_round": 1000,
    }


def save_model(model, target_column):
    model_path = MODEL_FOLDER / f"lgm_model_{target_column}.bin"
    model.save_model(model_path)


def column_name_df_preds(model_name: str) -> str:
    return "pred_" + model_name.split("_")[-1].split(".bin")[0]


def read_and_normalize_data(file_path: str) -> pd.DataFrame:
    pred_df = pd.read_csv(file_path, sep=",")
    for col in ["pred_Dump", "pred_Load"]:
        pred_df[col] = pred_df[col] / pred_df[col].max()
    return pred_df


def plot_data(ax, n_samples, data, label, marker, color, size, alpha=1.0) -> None:
    ax.scatter(
        n_samples, data, label=label, marker=marker, color=color, s=size, alpha=alpha
    )


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
