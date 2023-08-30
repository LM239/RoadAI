import joblib
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
import time
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
    fig_lc, ax_lc = plt.subplots(figsize=(8, 4))
    ax_lc.set_yscale("log")
    actions_taken_for_this_model = arg_track_performance().action
    dataset = arg_track_performance().data_set

    # for idx, target_column in enumerate(y_train.columns):
    # train = lgbm.Dataset(X_train, y_train[target_column])
    # val = lgbm.Dataset(X_val, y_val[target_column])
    booster_record_eval = {}
    model = lgbm.LGBMClassifier(
        class_weight={"Load": 2000, "Dump": 2000, "Driving": 1}, verbose=-1
    )
    t0 = time.perf_counter()
    model = model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val),
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
        ax=ax_lc,
        dataset=dataset,
    )
    fig_fi = lgbm.plot_importance(model).figure
    fig_fi.tight_layout()
    fig_fi.savefig("data/ml_model_data/pngs/feature_importance.png")

    training_time = time.perf_counter() - t0
    # track the last val_loss to evaluate performance
    write_performance_to_txt_file(
        training_time,
        actions_taken_for_this_model,
        dataset,
        booster_record_eval["Val"]["multi_logloss"][-1],
    )
    save_model(model)

    fig_lc.tight_layout()
    fig_lc.savefig(f"{FOLDER_NAME}/pngs/learning_curve.png")


def _load_and_predict() -> None:
    """
    Load the two models and predict load and dump. Store the preds with the empirical data
    """
    dataset = arg_track_performance().data_set
    X_train, X_test, y_train, y_test = split_data_into_training_and_testing(dataset)
    pred_df_testing = pd.concat([X_test, y_test], axis=1)
    pred_df_training = pd.concat([X_train, y_train], axis=1)

    loaded_model = joblib.load(f"{FOLDER_NAME}/models/lgm_model.bin")

    # this is the order of the output matrix
    driving_label, dump_label, load_label = loaded_model.classes_

    pred_training = loaded_model.predict_proba(X_train)
    pred_testing = loaded_model.predict_proba(X_test)

    pred_df_training[f"proba_{driving_label}"] = pred_training[:, 0]
    pred_df_training[f"proba_{dump_label}"] = pred_training[:, 1]
    pred_df_training[f"proba_{load_label}"] = pred_training[:, 2]

    pred_df_testing[f"proba_{driving_label}"] = pred_testing[:, 0]
    pred_df_testing[f"proba_{dump_label}"] = pred_testing[:, 1]
    pred_df_testing[f"proba_{load_label}"] = pred_testing[:, 2]

    # store load and dump avg. probability
    (load_proba, dump_proba, driving_proba) = get_avg_probabilities(pred_df_testing)
    # save to file
    write_proba_score_test_data(load_proba, dump_proba, driving_proba)

    save_pred_df(pred_df_testing, "testing")
    save_pred_df(pred_df_training, "training")


# def _plot_pred_vs_empirical() -> None:
#     pred_df = read_and_normalize_data(
#         f"{FOLDER_NAME}/preds/preds_load_dump_testing.csv"
#     )
#     fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=False)
#     n_samples = range(len(pred_df))
#     dataset = arg_track_performance().data_set
#     # plot Load and Dump in seperate subfigures
#     for idx, label in enumerate(["Load", "Dump"]):
#         # plot prediction
#         plot_data(
#             axs[idx],
#             n_samples,
#             pred_df[f"pred_{label}"],
#             f"pred {label}",
#             "s",
#             "tab:red",
#             7,
#         )
#         # plot empirical
#         plot_data(
#             axs[idx], n_samples, pred_df[label], label, "o", "tab:blue", 7, alpha=0.5
#         )

#         axs[idx].legend()
#         axs[idx].set_title(f"{label}, training_file: {dataset.split('.csv')[0]}")
#         # axs[idx].set_ylim(0.8, 1.05)

#     fig.tight_layout()
#     fig.savefig(
#         f"{FOLDER_NAME}/pngs/empirical_vs_normalized_preds{dataset.split('.csv')[0]}.png"
#     )


if __name__ == "__main__":
    args = arg_track_performance()
    print("Action Argument:", args.action)
    print("Data Set:", args.data_set)

    _train_lightgbm_and_plot_metric()
    _load_and_predict()
    # _plot_pred_vs_empirical()
