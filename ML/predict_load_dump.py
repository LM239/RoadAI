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
    plot_metrics,
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
    dataset: str = "train_1_days_all_trucks_multi_new_feat.csv", action: str = "..."
) -> None:
    """
    Train and fit model to trainng data (80% of available data), train on both
    load and dump as lightgbm does not support multi-output regression
    """
    df_training = pd.read_csv(f"data/ml_model_data/training_data/{dataset}")

    # split again to get data to validate against during iterations
    X_train, X_val, y_train, y_val = split_data_into_training_and_validation(
        df_training
    )
    # predict both load and dump and save the models
    fig_lc, ax_lc = plt.subplots(figsize=(8, 4))
    ax_lc.set_yscale("log")

    # for idx, target_column in enumerate(y_train.columns):
    # train = lgbm.Dataset(X_train, y_train[target_column])
    # val = lgbm.Dataset(X_val, y_val[target_column])
    booster_record_eval = {}
    model = lgbm.LGBMClassifier(
        n_estimators=1000,
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

    plot_metrics(
        booster=booster_record_eval,
        metric=LightGBMParams.params["metric"],
        ax=ax_lc,
        dataset=dataset,
    )
    # save feature importances
    fig_fi = lgbm.plot_importance(model).figure
    fig_fi.tight_layout()
    fig_fi.savefig("data/ml_model_data/pngs/feature_importance.png")

    # plot split value histograms for each feature
    for feature in X_train.columns:
        plot_split_value_histogram(model, feature)

    training_time = time.perf_counter() - t0
    # track the last val_loss to evaluate performance
    write_performance_to_txt_file(
        training_time,
        action,
        dataset,
        booster_record_eval["Val"]["multi_logloss"][-1],
    )
    save_model(model)

    fig_lc.tight_layout()
    fig_lc.savefig(f"{FOLDER_NAME}/pngs/learning_curve.png")


def _load_and_predict(
    test_set: str = "test_1_days_all_trucks_multi_new_feat.csv",
) -> None:
    """
    Load the two models and predict load and dump. Store the preds with the empirical data
    """
    df_testing = pd.read_csv(f"data/ml_model_data/testing_data/{test_set}")

    loaded_model = joblib.load(f"{FOLDER_NAME}/models/lgm_model.bin")

    # this is the order of the output matrix
    driving_label, dump_label, load_label = loaded_model.classes_

    # pred_training = loaded_model.predict_proba(X_train)
    pred_testing: np.ndarray = loaded_model.predict_proba(
        df_testing.drop(["output_labels", "MachineID", "DateTime"], axis=1)
    )

    df_testing[f"proba_{driving_label}"] = pred_testing[:, 0]
    df_testing[f"proba_{dump_label}"] = pred_testing[:, 1]
    df_testing[f"proba_{load_label}"] = pred_testing[:, 2]

    # store load and dump avg. probability
    load_proba, dump_proba, driving_proba = get_avg_probabilities(df_testing)
    # save to file
    write_proba_score_test_data(load_proba, dump_proba, driving_proba)

    save_pred_df(df_testing, "testing")


if __name__ == "__main__":
    _train_lightgbm_and_plot_metric()
    _load_and_predict()
    # _plot_pred_vs_empirical()
