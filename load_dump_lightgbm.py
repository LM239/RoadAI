import helper_functions.dataloader as dataloader
from pathlib import Path
from pydantic import BaseModel
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from helper_functions.schemas import Machine
import geopy.distance
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgbm
import time
from lightgbm import early_stopping, record_evaluation, LGBMModel
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
)
from typing import Literal
from helper_functions.schemas import Position
import sys


class Stats(BaseModel):
    """
    A class used to represent statistics related to movement or travel over a day

    ...

    Attributes
    ----------
    day_speeds : list[float]
        a list containing floating-point numbers representing speeds for a particular day (default empty list)

    day_acceleration : list[float]
        a list containing floating-point numbers representing acceleration values for a specific day (default empty list)

    day_dists : list[float]
        a list containing floating-point numbers representing distances traveled in a specific day (default empty list)

    day_times : list[datetime]
        a list containing datetime objects representing timestamps (default empty list)

    """

    day_speeds: list[float] = []
    day_acceleration: list[float] = []
    day_dists: list[float] = []
    day_times: list[datetime] = []


class PrepareMachineData:
    """
    A class used to prepare machine data for machine learning analysis.

    ...

    Attributes
    ----------
    machine : Machine
        the Machine object that contains all the machine-related data.
    stats : Stats
        the Stats object that will store various statistics calculated from the machine data.

    Methods
    -------
    get_speed_and_acceleration()
        Calculates speed and acceleration for every possible position using a forward derivative scheme

    construct_df_for_training(group_size: int)
        Generates a DataFrame with machine learning features and labels, grouped by the specified size.
    """

    def __init__(self, machine_data: Machine):
        self.machine = machine_data
        self.stats = Stats()

    def get_speed_and_acceleration(self) -> None:
        self.stats.day_times = [point.timestamp for point in self.machine.all_positions]

        # get statistics for all positions
        for i in range(1, len(self.machine.all_positions) - 1):
            next_pos = self.machine.all_positions[i + 1]
            current_pos = self.machine.all_positions[i]
            prev_pos = self.machine.all_positions[i - 1]

            # Meters driven since last timestamp
            meters_driven_cur_prev, time_cur_prev = calculate_distance_and_time(
                current_pos, prev_pos
            )
            meters_driven_next_cur, time_next_cur = calculate_distance_and_time(
                next_pos, current_pos
            )
            # add speed and accelerations
            try:
                speed_ms_i_minus_1 = meters_driven_cur_prev / time_cur_prev
                speed_ms_i = meters_driven_next_cur / time_next_cur
                acceleration_i_minus_1 = (
                    speed_ms_i - speed_ms_i_minus_1
                ) / time_cur_prev
            except ZeroDivisionError:
                # rows (duplicates) containing NaN are removed prior to training
                speed_ms_i_minus_1 = np.nan
                acceleration_i_minus_1 = np.nan

            self.stats.day_acceleration.append(acceleration_i_minus_1)
            self.stats.day_speeds.append(speed_ms_i_minus_1)

    def construct_df_for_training(self, group_size) -> pd.DataFrame:
        load_times = [load.timestamp for load in self.machine.all_loads]
        dump_times = [dump.timestamp for dump in self.machine.all_dumps]
        uncertainty = [point.uncertainty for point in self.machine.all_positions]
        latitude = [point.lat for point in self.machine.all_positions]
        longitude = [point.lon for point in self.machine.all_positions]
        lat1_minus_lat0 = [
            latitude[i] - latitude[i - 1] for i in range(1, len(latitude))
        ]
        lon1_minus_lon0 = [
            longitude[i] - longitude[i - 1] for i in range(1, len(longitude))
        ]
        speed_north_south = np.zeros_like(np.array(latitude))
        speed_east_west = np.zeros_like(np.array(latitude))

        for idx in range(1, len(latitude) - 1):
            try:
                total_seconds = (
                    self.stats.day_times[idx] - self.stats.day_times[idx - 1]
                ).total_seconds()
                speed_east_west[idx - 1] = (
                    longitude[idx] - longitude[idx - 1]
                ) / total_seconds
                speed_north_south[idx - 1] = (
                    latitude[idx] - latitude[idx - 1]
                ) / total_seconds
            except ZeroDivisionError:
                speed_east_west[idx - 1] = np.nan
                speed_north_south[idx - 1] = np.nan

        output_labels = np.zeros_like(self.stats.day_times)
        for i in range(len(output_labels)):
            current_time = self.stats.day_times[i]  # The current timestamp
            if current_time in load_times:
                output_labels[i] = "Load"
            elif current_time in dump_times:
                output_labels[i] = "Dump"
            else:
                output_labels[i] = "Driving"

        # Add values to compensate for the different lengths
        for _ in range(2):
            self.stats.day_speeds.append(np.nan)
            self.stats.day_acceleration.append(np.nan)
        lat1_minus_lat0.append(lat1_minus_lat0[-1])
        lon1_minus_lon0.append(lon1_minus_lon0[-1])
        # construct the df
        df = pd.DataFrame(
            {
                "MachineID": [self.machine.machine_id] * len(self.stats.day_times),
                "DateTime": self.stats.day_times,
                "Speed": self.stats.day_speeds,
                "Acceleration": self.stats.day_acceleration,
                "Latitude": latitude,
                "Longitude": longitude,
                "Uncertainty": uncertainty,
                "Lat1_minus_lat0": lat1_minus_lat0,
                "Lon1_minus_lon0": lon1_minus_lon0,
                "speed_north_south": speed_north_south,
                "speed_east_west": speed_east_west,
                "output_labels": output_labels,
            }
        )

        # filter df to remove the rows after the last dump
        last_row = df.query('output_labels == "Dump"').index[-1]
        df = df.loc[:last_row]

        # Merge 'group_size' data points into one. Start making an empty DataFrame
        result_df = pd.DataFrame()

        # Transform original DataFrame (df) into a new DataFrame (result_df) with additional features
        for i in range(group_size):
            sub_df = df.iloc[i::group_size]
            sub_df = sub_df.reset_index(drop=True)
            new_column_names = {}
            for column_name in df.columns:
                new_column_names[column_name] = f"{column_name}_{i}"

            # Rename the columns with the updated names
            sub_df.rename(columns=new_column_names, inplace=True)

            if i == 0:
                result_df = sub_df
            else:
                result_df = pd.concat([result_df, sub_df], axis=1)

        def custom_function(row):
            for col in result_df.columns:
                if "output_labels" in col:
                    if row[col] == "Load":
                        return "Load"
                    elif row[col] == "Dump":
                        return "Dump"
            return "Driving"

        result_df["output_labels"] = result_df.apply(custom_function, axis=1)
        result_df.rename(columns={"DateTime_0": "DateTime"}, inplace=True)
        result_df.rename(columns={"MachineID_0": "MachineID"}, inplace=True)
        for i in range(group_size):
            result_df = result_df.drop(f"output_labels_{i}", axis=1)
            if i != 0:
                result_df = result_df.drop(f"DateTime_{i}", axis=1)
                result_df = result_df.drop(f"MachineID_{i}", axis=1)

        return result_df


def calculate_distance_and_time(
    cur_pos: Position, prev_pos: Position
) -> tuple[float, float]:
    """
    The function calculates the distance and time between two positions.

    :param cur_pos: The current position, represented as a `Position` object
    :type cur_pos: Position
    :param prev_pos: The `prev_pos` parameter represents the previous position, which includes the
    latitude, longitude, and timestamp of the previous location
    :type prev_pos: Position
    :return: the distance (m) and time (s) between two positions.
    """
    distance = geopy.distance.geodesic(
        (cur_pos.lat, cur_pos.lon), (prev_pos.lat, prev_pos.lon)
    ).m
    time = (cur_pos.timestamp - prev_pos.timestamp).total_seconds()
    return distance, time


def load_training_csv_files(
    name_of_data_set: str,
) -> pd.DataFrame:
    """
    The function `load_training_csv_files` loads a CSV file as a pandas DataFrame from a specified
    directory.

    :param name_of_data_set: The parameter `name_of_data_set` is a string that represents the name of
    the CSV file that you want to load as a training dataset
    :type name_of_data_set: str
    :return: a pandas DataFrame.
    """
    return pd.read_csv(
        f"data/ml_model_data/training_data/{name_of_data_set}", delimiter=","
    )


def split_data_into_training_and_validation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    The function `split_data_into_training_and_validation` divides the given DataFrame into feature and label
    DataFrames, and further splits those into training and testing sets.

    :param df: The input DataFrame containing the features and labels.
    :type df: pd.DataFrame

    :return: A tuple containing four DataFrames: X_train, X_test, y_train, y_test.
    - X_train: Features for the training set
    - X_test: Features for the testing set
    - y_train: Labels corresponding to X_train
    - y_test: Labels corresponding to X_test
    """
    X, y = (
        df.drop(["MachineID", "output_labels", "DateTime"], axis=1),
        df["output_labels"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40
    )
    return (X_train, X_test, y_train, y_test)


def get_learning_curve(
    booster: dict | LGBMModel,
    metric: str | None,
    ax,
    dataset: str,
) -> None:
    """
    The function `plot_learning_curve` plots the learning curve using LightGBM's built-in plotting
    functionality for the specified metric and dataset.

    :param booster: The trained LightGBM model or dictionary containing information about the booster.
    :type booster: dict or LGBMModel

    :param metric: The metric name to plot. Could be `None`, in which case it will use the first metric it finds.
    :type metric: str or None

    :param ax: The matplotlib axis object on which to plot the learning curve.
    :type ax: matplotlib axis object

    :param dataset: The dataset for which to plot the learning curve (e.g., "Train", "Validation").
    :type dataset: str

    :return: None. The function directly plots the learning curve on the provided axis object.
    """
    lgbm.plot_metric(
        booster=booster,
        metric=metric,
        ax=ax,
        grid=True,
        title=f"Learning curve for dataset: {dataset}",
        ylabel="Multi logloss",
    )


class CustomLightgbmParams:
    def __init__(self, metric="multi_logloss", n_estimators=10000):
        self.metric = metric
        self.n_estimators = n_estimators


class LoadDumpLightGBM:
    """
    A class used for machine learning analysis of load and dump data
    with LightGBM.

    Attributes
    ----------
    nb_days : int or Literal["all"]
        Number of days of data to use for model training and testing.
        Raises ValueError if int(nb_days) exceeds the available number of days accessable
    group_size : int
        Size of each data group. A larger value means the model will predict on a larger timeframe
    starting_from : int
        Index to start from in the list of available days.
    work_dir : str
        Directory where the model data will be saved.
    gps_data_dir : str
        Directory where the GPS data is stored.
    training_data_name : str
        The name of the training data file.
    test_data_name : str
        The name of the test data file.
    lgbm_custom_params : CustomLightgbmParams
        Custom parameters for LightGBM training.
    booster_record_eval : dict
        Dictionary to record evaluation metrics during model training.

    Methods
    -------
    load_data()
        Load the data for model training and testing.
    fit_model(stopping_rounds: int = 2)
        Fit the LightGBM model.
    plot_feature_importances()
        Plot and save the feature importances of the trained model.
    plot_learning_curve()
        Plot and save the learning curve of the trained model.
    predict_and_save_csv()
        Make predictions on the test set.
    plot_statistics()
        Plot precision, recall and f1-score into 2d plot as a matrix
    plot_confusion_matrix()
        Plot and save the confusion matrix of the model.
    """

    def __init__(
        self,
        group_size: int = 5,
        nb_days: int | Literal["all"] = 1,
        starting_from: int = 0,
        gps_data_dir: str = "data/GPSData",
        work_dir: str = "data/ml_model_data",
        machine_type: Literal["Truck", "Dumper"] = "Truck",
    ) -> None:
        #  check if folder structure is ok, verify nb_days
        self._check_folder_structure(gps_data_dir)
        self._validate_nb_days(nb_days, gps_data_dir)

        print("Initializing class:")
        print("----------------------")
        print("Data over: ", nb_days, "days.")
        print("Merging ", group_size, " consecutive timestamps")
        print("Model applies on machine type:", machine_type)
        print("All data will be saved to the automatically created path: ", work_dir)
        self.nb_days = (
            nb_days
            if isinstance(nb_days, int)
            else len(
                os.listdir(gps_data_dir + "/trips")
            )  # 'else' is executed if nb_days is a valid string ('all')
        )
        self.group_size = group_size
        self.starting_from = starting_from
        self.work_dir = work_dir
        self.machine_type = machine_type
        self.work_dir_day = f"{work_dir}/class_data_{self.nb_days}_days_from_day_{self.starting_from}_{self.machine_type}"
        self.gps_data_dir = gps_data_dir
        self.training_data_name = "training_and_validation_data"
        self.test_data_name = "testing_data"
        self.lgbm_custom_params = CustomLightgbmParams()

    def _check_folder_structure(self, gps_data_dir: str):
        for folder in ["trips", "tripsInfo"]:
            folder_path = os.path.join(gps_data_dir, folder)
            if not os.path.isdir(folder_path):
                raise FileNotFoundError(
                    f"Please have folders 'trips' and 'tripsInfo' located in {gps_data_dir}"
                )
        print("Folders 'trips' and 'tripsInfo' correctly set up")

    def _validate_nb_days(self, nb_days: int | Literal["all"], gps_data_dir: str):
        if isinstance(nb_days, int):
            if nb_days > len(os.listdir(os.path.join(gps_data_dir, "trips"))):
                raise ValueError(
                    f"The 'nb_days' parameter ({nb_days}) cannot be greater than the number"
                    f"of days in GPSData/trips: ({len(os.listdir(os.path.join(gps_data_dir, 'trips')))})."
                )
        elif isinstance(nb_days, str):
            if nb_days.lower() != "all":
                raise ValueError("The string value for 'nb_days' must be 'all'.")
        else:
            raise ValueError(
                "The 'nb_days' parameter must be an integer or the string 'all'."
            )

    def load_data(self) -> None:
        self.days = [
            csv_file.split(".csv")[0]
            for csv_file in os.listdir(f"{self.gps_data_dir + '/trips'}")
        ]
        print("Start at day ", self.days[self.starting_from])
        print("For machine type: ", self.machine_type)

        df_training_all = pd.DataFrame()
        df_testing_all = pd.DataFrame()

        # Make sure original console behaviour is stored
        # Pandas raises message for day 03-10-2022

        for day in tqdm(
            self.days[self.starting_from : self.starting_from + self.nb_days]
        ):
            trip = dataloader.TripsLoader(day)
            for _, machine in trip._machines.items():
                if machine.machine_type == self.machine_type:
                    automated_for_given_machine = PrepareMachineData(machine)
                    automated_for_given_machine.get_speed_and_acceleration()
                    try:
                        df_vehicle = (
                            automated_for_given_machine.construct_df_for_training(
                                self.group_size
                            )
                        )
                    except IndexError:
                        # if a trip only has one timestamp, do not use this trip
                        continue

                    X, y = (
                        df_vehicle.drop(["output_labels"], axis=1),
                        df_vehicle["output_labels"],
                    )
                    # Each vehicle should be represented 20% for each day in the test data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=40
                    )

                    df_training = pd.concat([X_train, y_train], axis=1).sort_values(
                        by="DateTime"
                    )

                    df_training_all = pd.concat([df_training_all, df_training], axis=0)
                    df_testing = pd.concat([X_test, y_test], axis=1).sort_values(
                        by="DateTime"
                    )
                    df_testing_all = pd.concat([df_testing_all, df_testing], axis=0)

            df_training_all.dropna(inplace=True)
            df_testing_all.dropna(inplace=True)

            Path(self.work_dir_day).mkdir(parents=True, exist_ok=True)
            df_training_all.to_csv(
                f"{self.work_dir_day}/{self.training_data_name}.csv",
                sep=",",
                index=False,
            )
            df_testing_all.to_csv(
                f"{self.work_dir_day}/{self.test_data_name}.csv",
                sep=",",
                index=False,
            )

    def fit_model(self, stopping_rounds: int = 50) -> None:
        df_training = pd.read_csv(f"{self.work_dir_day}/{self.training_data_name}.csv")

        X_train, X_val, y_train, y_val = split_data_into_training_and_validation(
            df_training
        )

        count_series = df_training["output_labels"].value_counts()
        model = lgbm.LGBMClassifier(
            **self.lgbm_custom_params.__dict__,
            class_weight={
                "Driving": 1,
                "Load": count_series["Driving"] / count_series["Load"],
                "Dump": count_series["Driving"] / count_series["Dump"],
            },
            verbose=-1,
        )
        self.booster_record_eval = {}
        t0 = time.perf_counter()
        model = model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_train, y_train),
                (X_val, y_val),
            ],
            eval_metric=self.lgbm_custom_params.metric,
            eval_names=["Train", "Val"],
            callbacks=[
                early_stopping(stopping_rounds=stopping_rounds),
                record_evaluation(self.booster_record_eval),
            ],
        )

        # Save training time and validaton data error at termination
        with open(f"{self.work_dir}/track_results_days_and_machines.txt", "a") as f:
            f.write(
                f"\n\n Early stopping: {stopping_rounds}\n"
                f"Number of iterations: {self.lgbm_custom_params.n_estimators} \n"
                f"Training time: {round((time.perf_counter() - t0),3)} s\n"
                f"Data set path: {Path(self.work_dir_day) / self.training_data_name}.\n"
                f"Validation multi-logloss at termination: {round(self.booster_record_eval['Val']['multi_logloss'][-1],5)}\n"
                f"Machine type: {self.machine_type} \n"
                f"Best iteration: {model.best_iteration_} \n \n"
            )
            f.flush()

        joblib.dump(model, f"{self.work_dir_day}/lgbm_model.bin")

    def plot_feature_importances(self) -> None:
        # Save feature importances as png
        fig_fi = lgbm.plot_importance(
            joblib.load(f"{self.work_dir_day}/lgbm_model.bin"),
            figsize=(15, 10),
        ).figure
        fig_fi.tight_layout()
        fig_fi.savefig(f"{self.work_dir_day}/feature_importance.png")

    def plot_learning_curve(self) -> None:
        # Save learning curve as png
        fig_lc, ax_lc = plt.subplots(figsize=(6, 4))
        ax_lc.set_yscale("log")
        get_learning_curve(
            booster=self.booster_record_eval,
            metric=self.lgbm_custom_params.metric,
            ax=ax_lc,
            dataset=f"{self.training_data_name}",
        )
        fig_lc.tight_layout()
        fig_lc.savefig(f"{self.work_dir_day}/learning_curve.png")

    def predict_and_save_csv(self) -> None:
        df_testing = pd.read_csv(f"{self.work_dir_day}/{self.test_data_name}.csv")

        loaded_model = joblib.load(f"{self.work_dir_day}/lgbm_model.bin")

        # this is the order of the output matrix
        driving_label, dump_label, load_label = loaded_model.classes_

        pred_testing_probas: np.ndarray = loaded_model.predict_proba(
            df_testing.drop(["output_labels", "MachineID", "DateTime"], axis=1)
        )
        pred_testing_labels: np.ndarray = loaded_model.predict(
            df_testing.drop(["output_labels", "MachineID", "DateTime"], axis=1)
        )

        df_testing[f"proba_{driving_label}"] = pred_testing_probas[:, 0]
        df_testing[f"proba_{dump_label}"] = pred_testing_probas[:, 1]
        df_testing[f"proba_{load_label}"] = pred_testing_probas[:, 2]
        df_testing["predicted_class"] = pred_testing_labels
        df_testing.to_csv(
            f"{self.work_dir_day}/preds_on_test_data.csv",
            sep=",",
            index=False,
        )

    def plot_statistics(self) -> None:
        df_preds = pd.read_csv(
            f"{self.work_dir_day}/preds_on_test_data.csv",
            sep=",",
            usecols=[
                "output_labels",
                "predicted_class",
            ],
        )
        y_true = df_preds["output_labels"]
        y_pred = df_preds["predicted_class"]
        class_report: dict = classification_report(y_true, y_pred, output_dict=True)  # type: ignore

        # assuming labels are keys in class_report with metrics as values
        labels = ["Driving", "Dump", "Load"]
        metrics = ["precision", "recall", "f1-score"]
        data = np.zeros((len(labels), len(metrics)))

        for idx1, label in enumerate(labels):
            for idx2, metric in enumerate(metrics):
                data[idx1][idx2] = round(class_report[label][metric], 3)

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.matshow(data, cmap="Blues", fignum=0)  # plot in 'ax'
        plt.xticks(np.arange(len(metrics)), metrics)
        plt.yticks(np.arange(len(labels)), labels)
        plt.colorbar(cmap="Blues")

        for i in range(len(labels)):
            for j in range(len(metrics)):
                plt.text(j, i, str(data[i, j]), va="center", ha="center")

        plt.xlabel("Metrics")
        plt.ylabel("Labels")
        plt.tight_layout()

        plt.savefig(f"{self.work_dir_day}/statistics.png")

    def plot_confusion_matrix(self) -> None:
        df_preds = pd.read_csv(
            f"{self.work_dir_day}/preds_on_test_data.csv",
            sep=",",
            usecols=["output_labels", "predicted_class"],
        )
        y_true = df_preds["output_labels"]
        y_pred = df_preds["predicted_class"]
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", ax=ax)
        fig.tight_layout()
        fig.savefig(f"{self.work_dir_day}/confusion_matrix.png")
        plt.show()
