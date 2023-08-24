import lightgbm as lgbm
from lightgbm import early_stopping, record_evaluation
from sklearn.model_selection import train_test_split
from data_preprocessing_ml import (
    split_data_into_training_and_testing,
    data_set_to_consider,
)
from func_helpers_ml import (
    FOLDER_NAME,
    column_name_df_preds,
    save_pred_df,
    plot_metrics,
    LightGBMParams,
)
import matplotlib.pyplot as plt
import pandas as pd
import os


def _train_lightgbm() -> None:
    """
    Train and fit model to trainng data (80% of available data), train on both
    load and dump as lightgbm does not support multi-output regression
    """
    X, _, y, _ = split_data_into_training_and_testing()  # type: ignore
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
        model.save_model(f"data/ml_model_data/models/lgm_model_{target_column}.bin")
    fig.tight_layout()
    fig.savefig(f"{FOLDER_NAME}/pngs/learning_curve.png")


def _load_and_predict() -> None:
    """
    Load the two models and predict load and dump. Store the preds with the empirical data
    """
    X_train, X_test, y_train, y_test = split_data_into_training_and_testing()  # type: ignore
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
    pred_df = pd.read_csv(f"{FOLDER_NAME}/preds/preds_load_dump_testing.csv", sep=",")
    # normalize
    for col in ["pred_Dump", "pred_Load"]:
        pred_df[col] = pred_df[col] / pred_df[col].max()

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=False)
    n_samples = range(len(pred_df))

    # plot Load and Dump in seperate subfigures
    for idx, label in enumerate(["Load", "Dump"]):
        # plot pred
        axs[idx].scatter(
            n_samples,
            pred_df[f"pred_{label}"],
            label=f"pred {label}",
            marker="s",
            color="tab:red",
            s=7,
        )
        # plot empirical
        axs[idx].scatter(
            n_samples,
            pred_df[label],
            label=label,
            alpha=0.5,
            marker="o",
            color="tab:blue",
            s=7,
        )
        axs[idx].legend()
        axs[idx].set_title(
            f"{label}, training_file: {data_set_to_consider().split('.csv')[0]}"
        )
        axs[idx].set_ylim(0.8, 1.05)

    fig.tight_layout()
    fig.savefig(
        f"{FOLDER_NAME}/pngs/empirical_vs_normalized_preds{data_set_to_consider().split('.csv')[0]}.png"
    )


if __name__ == "__main__":
    _train_lightgbm()
    _load_and_predict()
    _plot_pred_vs_empirical()
