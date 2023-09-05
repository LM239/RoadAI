import joblib
import lightgbm as lgbm
from lightgbm import early_stopping, record_evaluation
from sklearn.model_selection import train_test_split
from data_preprocessing import split_data_into_training_and_validation
from model_helpers import (
    FOLDER_NAME,
    get_avg_probabilities,
    plot_split_value_histogram,
    save_pred_df,
    plot_learning_curve,
    save_model,
    write_performance_to_txt_file,
    LightGBMParams,
    write_proba_score_test_data,
)
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _train_lightgbm_and_plot_metric(
    merged_timestamps,
    dataset: str = "train_20_days_all_trucks_all_data.csv",
    action: str = "...",
) -> None:
    """
    Train model on dataset of choice. To get a dataset, run automate_load_and_dump.ipynb
    The model is saved in data/ml_model_data/models
    The target variables are Dump, Load and Driving
    Learning curve, feature importance and other output are plotted and be found in  data/ml_model_data/pngs
    Small pieces of information about the training process and results are also found in data/ml_model_data/preds/track_performance.txt
    """
    df_training = pd.read_csv(f"data/ml_model_data/training_data/{dataset}")

    # split again to get data to validate against during iterations
    X_train, X_val, y_train, y_val = split_data_into_training_and_validation(
        merged_timestamps, df_training
    )
    # predict both load and dump and save the models
    fig_lc, ax_lc = plt.subplots(figsize=(8, 4))
    ax_lc.set_yscale("log")

    # for idx, target_column in enumerate(y_train.columns):
    # train = lgbm.Dataset(X_train, y_train[target_column])
    # val = lgbm.Dataset(X_val, y_val[target_column])
    booster_record_eval = {}
    model = lgbm.LGBMClassifier(
        n_estimators=10000,
        class_weight={"Load": 2000, "Dump": 2000, "Driving": 1},
        verbose=-1,
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

    plot_learning_curve(
        booster=booster_record_eval,
        metric=LightGBMParams.params["metric"],
        ax=ax_lc,
        dataset=dataset,
    )
    fig_lc.tight_layout()
    fig_lc.savefig(f"{FOLDER_NAME}/pngs/learning_curve.png")

    # plot feature importances
    fig_fi = lgbm.plot_importance(model).figure
    fig_fi.tight_layout()
    fig_fi.savefig("data/ml_model_data/pngs/feature_importance.png")

    # plot split value histograms for each feature
    for feature in X_train.columns:
        plot_split_value_histogram(model, feature)

    # save training time, val_error at termination
    write_performance_to_txt_file(
        time.perf_counter() - t0,
        action,  # ignore this, only if you run from terminal and add arguments with argparse
        dataset,
        booster_record_eval["Val"]["multi_logloss"][-1],
    )
    save_model(model)


def _load_and_predict(
    merged_timestamps: bool, test_set: str = "test_20_days_all_trucks_all_data.csv"
) -> None:
    """
    Load model and predict on unseen data
    """
    df_testing = pd.read_csv(f"data/ml_model_data/testing_data/{test_set}")

    loaded_model = joblib.load(f"{FOLDER_NAME}/models/lgm_model.bin")

    # this is the order of the output matrix
    driving_label, dump_label, load_label = loaded_model.classes_
    labels_to_drop = [
        [
            "output_labels",
            "MachineID",
            "n_rows_merged",
            "DateTime_min",
            "DateTime_max",
            "DateTime_mean",
        ]
        if merged_timestamps
        else ["output_labels", "MachineID", "DateTime"]
    ]

    # pred_training = loaded_model.predict_proba(X_train)
    pred_testing: np.ndarray = loaded_model.predict_proba(
        df_testing.drop(
            labels_to_drop[0],
            axis=1,
        )
    )

    df_testing[f"proba_{driving_label}"] = pred_testing[:, 0]
    df_testing[f"proba_{dump_label}"] = pred_testing[:, 1]
    df_testing[f"proba_{load_label}"] = pred_testing[:, 2]
    save_pred_df(df_testing, "testing")

    ############ the same piece of info can be found in preds/probabilities #############
    # store load and dump avg. probability
    load_proba, dump_proba, driving_proba = get_avg_probabilities(df_testing)
    # save to file
    write_proba_score_test_data(load_proba, dump_proba, driving_proba)


if __name__ == "__main__":
    _train_lightgbm_and_plot_metric(merged_timestamps=False)
    _load_and_predict(merged_timestamps=False)
