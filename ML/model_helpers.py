import pandas as pd
from lightgbm import LGBMModel
import numpy as np
import joblib
import lightgbm as lgbm
from pathlib import Path
import argparse

FOLDER_NAME = "data/ml_model_data"
MODEL_FOLDER = Path(f"{FOLDER_NAME}/models")


class LightGBMParams:
    params = {
        "boosting_type": "gbdt",
        "metric": "multi_logloss",
        "num_leaves": 31,
        "learning_rate": 0.5,
        "feature_fraction": 1,
        "num_boost_round": 1000,
    }


def save_model(model):
    model_path = MODEL_FOLDER / "lgm_model.bin"
    joblib.dump(model, model_path)


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
    ax,
    dataset: str,
) -> None:
    lgbm.plot_metric(
        booster=booster,
        metric=metric,
        ax=ax,
        grid=True,
        title=f"Learning curve {dataset}",
        ylabel="Multi logloss",
    )


def plot_split_value_histogram(model: LGBMModel, feature: str) -> None:
    fig_split_hist = lgbm.plot_split_value_histogram(model, feature).figure
    fig_split_hist.tight_layout()
    fig_split_hist.savefig(f"data/ml_model_data/pngs/split_value_hist_{feature}.png")


def write_performance_to_txt_file(
    training_time: float,
    action_taken: str,
    data_set: str,
    val_log_loss: float,
) -> None:
    with open("data/ml_model_data/preds/track_performance.txt", "a") as f:
        f.write(
            f"...\nTraining time: {training_time} s\nChanges: {action_taken}.\nData set: {data_set}.\nValidation multi logloss: {val_log_loss}\n"
        )


def get_avg_probabilities(df_pred: pd.DataFrame) -> tuple[float, float, float]:
    # filter to return only rows where we have loads and dumps
    true_loads_rows = df_pred.loc[df_pred["output_labels"] == "Load", "proba_Load"]
    true_dumps_rows = df_pred.loc[df_pred["output_labels"] == "Dump", "proba_Dump"]
    true_driving_rows = df_pred.loc[
        df_pred["output_labels"] == "Driving", "proba_Driving"
    ]
    # the length of both true_loads and true_dumps corresponds to the sum

    # we calculate the average probability
    load_proba = true_loads_rows.sum() / len(true_loads_rows)
    dump_proba = true_dumps_rows.sum() / len(true_dumps_rows)
    driving_proba = true_driving_rows.sum() / len(true_driving_rows)

    return (load_proba, dump_proba, driving_proba)


def write_proba_score_test_data(
    load_proba: float, dump_proba: float, driving_proba: float
) -> None:
    """
    Make sure probabilities are of order (load_proba, dump_proba)
    """

    with open("data/ml_model_data/preds/track_performance.txt", "a") as f:
        f.write(
            f"------------------\nLoad avg. proba: {load_proba}\nDump avg. proba {dump_proba}...\nDriving proba: {driving_proba}\n\n\n"
        )


def arg_track_performance():
    parser = argparse.ArgumentParser(
        description="Explain the changes applied for this model compared to last. Did you change params? Dataset?"
    )
    parser.add_argument(
        "--action",
        type=str,
        default="Some action",
        help="Explain the changes applied for this model compared to last. Did you change params? Dataset?",
    )
    parser.add_argument(
        "--data_set",
        type=str,
        default="default_dataset.csv",
        help="Specify the data set to be used.",
    )
    return parser.parse_args()
