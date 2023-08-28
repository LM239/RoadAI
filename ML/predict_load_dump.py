import lightgbm as lgbm
from lightgbm import early_stopping, record_evaluation
from sklearn.model_selection import train_test_split
from data_preprocessing import split_data_into_training_and_testing
from model_helpers import (
    FOLDER_NAME,
    column_name_df_preds,
    get_avg_probabilities,
    plot_data,
    read_and_normalize_data,
    save_pred_df,
    plot_metrics,
    save_model,
    write_performance_to_txt_file,
    LightGBMParams,
    write_proba_score_test_data,
)
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse


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


def _train_lightgbm_and_plot_metric() -> None:
    """
    Train and fit model to trainng data (80% of available data), train on both
    load and dump as lightgbm does not support multi-output regression
    """
    dataset = arg_track_performance().data_set
    X, _, y, _ = split_data_into_training_and_testing(dataset)  # type: ignore
    # split again to get data to validate against during iterations
    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=39)
    # predict both load and dump and save the models
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    actions_taken_for_this_model = arg_track_performance().action
    dataset = arg_track_performance().data_set
    for idx, target_column in enumerate(y_train.columns):
        # train = lgbm.Dataset(X_train, y_train[target_column])
        # val = lgbm.Dataset(X_val, y_val[target_column])
        booster_record_eval = {}
        model = lgbm.LGBMClassifier(class_weight={False: 1, True: 20})
        # print(type(y_train[target_column].values.ravel()))
        model = model.fit(
            X_train,
            y_train[target_column].values.ravel(),
            eval_set=[
                (X_train, y_train[target_column].values.ravel()),
                (X_val, y_val[target_column].values.ravel()),
            ],
            eval_metric=LightGBMParams.params["metric"],
            eval_names=["Train", "Val"],
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

        # track the last val_loss to evaluate performance
        write_performance_to_txt_file(
            actions_taken_for_this_model,
            dataset,
            booster_record_eval["Val"]["binary_logloss"][-1],
        )
        save_model(model, target_column)

    fig.tight_layout()
    fig.savefig(f"{FOLDER_NAME}/pngs/learning_curve.png")


def _load_and_predict() -> None:
    """
    Load the two models and predict load and dump. Store the preds with the empirical data
    """
    dataset = arg_track_performance().data_set
    X_train, X_test, y_train, y_test = split_data_into_training_and_testing(dataset)
    pred_df_testing = pd.concat([X_test, y_test], axis=1)
    pred_df_training = pd.concat([X_train, y_train], axis=1)

    model_names = [
        model for model in os.listdir(f"{FOLDER_NAME}/models") if model.endswith(".bin")
    ]
    for model_name in model_names:
        loaded_model = lgbm.Booster(model_file=f"{FOLDER_NAME}/models/{model_name}")

        pred_training = loaded_model.predict(X_train)
        # label columns as pred_Load and pred_dump
        pred_df_training[column_name_df_preds(model_name)] = pred_training
        pred = loaded_model.predict(X_test)
        pred_df_testing[column_name_df_preds(model_name)] = pred

    # store load and dump avg. probability
    (
        load_proba,
        dump_proba,
        incorrect_load_proba,
        incorrect_dump_proba,
    ) = get_avg_probabilities(pred_df_testing)
    # save to file
    write_proba_score_test_data(
        load_proba, dump_proba, incorrect_load_proba, incorrect_dump_proba
    )

    save_pred_df(pred_df_testing, "testing")
    save_pred_df(pred_df_training, "training")


def _plot_pred_vs_empirical() -> None:
    pred_df = read_and_normalize_data(
        f"{FOLDER_NAME}/preds/preds_load_dump_testing.csv"
    )
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=False)
    n_samples = range(len(pred_df))
    dataset = arg_track_performance().data_set
    # plot Load and Dump in seperate subfigures
    for idx, label in enumerate(["Load", "Dump"]):
        # plot prediction
        plot_data(
            axs[idx],
            n_samples,
            pred_df[f"pred_{label}"],
            f"pred {label}",
            "s",
            "tab:red",
            7,
        )
        # plot empirical
        plot_data(
            axs[idx], n_samples, pred_df[label], label, "o", "tab:blue", 7, alpha=0.5
        )

        axs[idx].legend()
        axs[idx].set_title(f"{label}, training_file: {dataset.split('.csv')[0]}")
        # axs[idx].set_ylim(0.8, 1.05)

    fig.tight_layout()
    fig.savefig(
        f"{FOLDER_NAME}/pngs/empirical_vs_normalized_preds{dataset.split('.csv')[0]}.png"
    )


if __name__ == "__main__":
    args = arg_track_performance()
    print("Action Argument:", args.action)
    print("Data Set:", args.data_set)

    _train_lightgbm_and_plot_metric()
    _load_and_predict()
    _plot_pred_vs_empirical()
