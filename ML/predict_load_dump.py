import lightgbm as lgbm
from lightgbm import early_stopping, record_evaluation
from sklearn.model_selection import train_test_split
from data_preprocessing import split_data_into_training_and_testing
from model_helpers import (
    FOLDER_NAME,
    column_name_df_preds,
    plot_data,
    read_and_normalize_data,
    save_pred_df,
    plot_metrics,
    save_model,
    LightGBMParams,
)
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Choose dataset.")
    parser.add_argument(
        "--data_set",
        type=str,
        default="2_days_1_vehicle.csv",
        help="Data set to consider (data/ml_model_data/training_data)",
    )
    return parser.parse_args()


def _train_lightgbm() -> None:
    """
    Train and fit model to trainng data (80% of available data), train on both
    load and dump as lightgbm does not support multi-output regression
    """
    dataset = get_args().data_set
    X, _, y, _ = split_data_into_training_and_testing(dataset)  # type: ignore
    # split again to get data to validate against during iterations
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=39)
    # predict both load and dump and save the models
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    for idx, target_column in enumerate(y_train.columns):
        train = lgbm.Dataset(X_train, y_train[target_column])
        val = lgbm.Dataset(X_val, y_val[target_column])
        booster_record_eval = {}
        model = lgbm.train(
            LightGBMParams.params,
            train,
            valid_sets=[train, val],
            valid_names=["Train", "Val"],
            callbacks=[
                early_stopping(stopping_rounds=2),
                record_evaluation(booster_record_eval),
            ],
        )
        plot_metrics(
            booster=booster_record_eval,
            metric=LightGBMParams.params["metric"],
            ax=axs[idx],
            load_or_dump=target_column,
        )
        save_model(model, target_column)

    fig.tight_layout()
    fig.savefig(f"{FOLDER_NAME}/pngs/learning_curve.png")


def _load_and_predict() -> None:
    """
    Load the two models and predict load and dump. Store the preds with the empirical data
    """
    dataset = get_args().data_set
    X_train, X_test, y_train, y_test = split_data_into_training_and_testing(dataset)
    pred_df_testing = pd.concat([X_test, y_test], axis=1)
    pred_df_training = pd.concat([X_train, y_train], axis=1)

    model_names = [
        model for model in os.listdir(f"{FOLDER_NAME}/models") if model.endswith(".bin")
    ]
    for model_name in model_names:
        loaded_model = lgbm.Booster(model_file=f"{FOLDER_NAME}/models/{model_name}")
        pred = loaded_model.predict(X_test)
        pred_df_testing[column_name_df_preds(model_name)] = pred

        pred_training = loaded_model.predict(X_train)
        pred_df_training[column_name_df_preds(model_name)] = pred_training

    save_pred_df(pred_df_testing, "testing")
    save_pred_df(pred_df_training, "training")


def _plot_pred_vs_empirical() -> None:
    pred_df = read_and_normalize_data(
        f"{FOLDER_NAME}/preds/preds_load_dump_testing.csv"
    )
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=False)
    n_samples = range(len(pred_df))
    dataset = get_args().data_set
    # plot Load and Dump in seperate subfigures
    for idx, label in enumerate(["Load", "Dump"]):
        plot_data(
            axs[idx],
            n_samples,
            pred_df[f"pred_{label}"],
            f"pred {label}",
            "s",
            "tab:red",
            7,
        )
        plot_data(
            axs[idx], n_samples, pred_df[label], label, "o", "tab:blue", 7, alpha=0.5
        )

        axs[idx].legend()
        axs[idx].set_title(f"{label}, training_file: {dataset.split('.csv')[0]}")
        axs[idx].set_ylim(0.8, 1.05)

    fig.tight_layout()
    fig.savefig(
        f"{FOLDER_NAME}/pngs/empirical_vs_normalized_preds{dataset.split('.csv')[0]}.png"
    )


if __name__ == "__main__":
    args = get_args()
    data_set = args.data_set
    print(f"Using data set: {data_set}")
    _train_lightgbm()
    _load_and_predict()
    _plot_pred_vs_empirical()
