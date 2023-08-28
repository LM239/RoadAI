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
        "metric": "binary_logloss",
        "num_leaves": 31,
        "learning_rate": 0.5,
        "feature_fraction": 1,
        "num_boost_round": 1000,
    }


def save_model(model, target_column):
    model_path = MODEL_FOLDER / f"lgm_model_{target_column}.bin"
    model.booster_.save_model(model_path)


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
        ylabel="Binary logloss",
    )


def write_performance_to_txt_file(
    action_taken: str, data_set: str, val_log_loss: float
) -> None:
    with open("data/ml_model_data/preds/track_performance.txt", "a") as f:
        f.write(
            f"...Changes: {action_taken}.\nData set: {data_set}.\nValidation logloss: {val_log_loss}\n"
        )


def get_avg_probabilities(df_pred: pd.DataFrame) -> tuple[float, float, float, float]:
    # filter to return only rows where we have loads and dumps
    true_loads_rows = df_pred.loc[df_pred["Load"] == True, "pred_Load"]
    true_dumps_rows = df_pred.loc[df_pred["Dump"] == True, "pred_Dump"]
    # the length of both true_loads and true_dumps corresponds to the sum
    # we calculate the average probability
    load_proba = true_loads_rows.sum() / len(true_loads_rows)
    dump_proba = true_dumps_rows.sum() / len(true_dumps_rows)

    # continue with calculating the probability of predicting dump and load incorrectly
    false_loads_rows = df_pred.loc[df_pred["Load"] == False, "pred_Load"]
    false_dumps_rows = df_pred.loc[df_pred["Dump"] == False, "pred_Dump"]
    # the length of both true_loads and true_dumps corresponds to the sum
    # we calculate the average probability
    incorrect_load_proba = false_loads_rows.sum() / len(false_loads_rows)
    incorrect_dump_proba = false_dumps_rows.sum() / len(false_dumps_rows)

    return (load_proba, dump_proba, incorrect_load_proba, incorrect_dump_proba)


def write_proba_score_test_data(
    load_proba: float,
    dump_proba: float,
    incorrect_load_proba: float,
    incorrect_dump_proba: float,
) -> None:
    """
    Make sure probabilities are of order (load_proba, dump_proba)
    """

    with open("data/ml_model_data/preds/track_performance.txt", "a") as f:
        f.write(
            f"Load avg. proba: {load_proba}\nDump avg. proba {dump_proba}...\nIncorrect load proba: {incorrect_load_proba}\nIncorrect dump proba: {incorrect_dump_proba}\n\n\n"
        )
