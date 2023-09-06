# %%
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
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from typing import Literal
from helper_functions.schemas import Position


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


# def get_avg_probabilities(df_pred: pd.DataFrame) -> tuple[float, float, float]:
#     """
#     The function `get_avg_probabilities` calculates the average probabilities for the "Load", "Dump",
#     and "Driving" labels in a given DataFrame.

#     :param df_pred: The parameter `df_pred` is a pandas DataFrame that contains the predictions made by
#     a model. It should have the following columns:
#     :type df_pred: pd.DataFrame
#     :return: a tuple of three floats representing the average probabilities for the "Load", "Dump", and
#     "Driving" labels in the given DataFrame.
#     """
#     # filter to return only rows where we have loads and dumps
#     true_loads_rows = df_pred.loc[df_pred["output_labels"] == "Load", "proba_Load"]
#     true_dumps_rows = df_pred.loc[df_pred["output_labels"] == "Dump", "proba_Dump"]
#     true_driving_rows = df_pred.loc[
#         df_pred["output_labels"] == "Driving", "proba_Driving"
#     ]

#     # Calculate average probabilties
#     load_proba = true_loads_rows.sum() / len(true_loads_rows)
#     dump_proba = true_dumps_rows.sum() / len(true_dumps_rows)
#     driving_proba = true_driving_rows.sum() / len(true_driving_rows)

#     return (load_proba, dump_proba, driving_proba)


# def write_proba_score_test_data(
#     load_proba: float, dump_proba: float, driving_proba: float
# ) -> None:
#     """
#     The function `write_proba_score_test_data` appends the average probabilities of three
#     classes ('Load', 'Dump', 'Driving') to a text file.

#     :param load_proba: The average probability score for the 'Load' class.
#     :type load_proba: float

#     :param dump_proba: The average probability score for the 'Dump' class.
#     :type dump_proba: float

#     :param driving_proba: The average probability score for the 'Driving' class.
#     :type driving_proba: float

#     :return: None. The function writes the average probabilities to a text file.
#     """
#     track_performance_file_path = Path("data/ml_model_data/preds/track_performance.txt")
#     track_performance_file_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(track_performance_file_path, "a") as f:
#         f.write(
#             f"------------------\nLoad avg. proba: {load_proba}\nDump avg. proba {dump_proba}...\nDriving proba: {driving_proba}\n\n\n"
#         )
#         f.flush()


class LoadDumpLightGBM:
    def __init__(
        self,
        group_size: int = 5,
        nb_days: int | Literal["all"] = 1,
        starting_from: int = 0,
        work_dir: str = "data/ml_model_data/class_data",
        gps_data_dir: str = "data/GPSData",
    ) -> None:
        if isinstance(nb_days, int):
            if nb_days > len(os.listdir(gps_data_dir + "/trips")):
                raise ValueError(
                    f"The 'nb_days' parameter ({nb_days}) cannot be greater than the number of days ({len(os.listdir(gps_data_dir+'/trips'))})."
                )
        elif isinstance(nb_days, str):
            if nb_days.lower() != "all":
                raise ValueError("The string value for 'nb_days' must be 'all'.")
        else:
            raise ValueError(
                "The 'nb_days' parameter must be an integer or the string 'all'."
            )

        print("Initializing class:")
        print("----------------------")
        print("Data over: ", nb_days, "days.")
        print("Merging ", group_size, " consecutive timestamps")
        print("All data saved to ", work_dir)
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
        self.gps_data_dir = gps_data_dir
        self.training_data_name = "my_train_from_class"
        self.test_data_name = "my_test_from_class"

        self.LightGBMParams = {
            "boosting_type": "gbdt",
            "metric": "multi_logloss",
            "num_leaves": 31,
            "learning_rate": 0.5,
            "feature_fraction": 1,
            "num_boost_round": 1000,
        }

    def load_data(self):
        self.days = [
            csv_file.split(".csv")[0]
            for csv_file in os.listdir(f"{self.gps_data_dir}/trips")
        ]
        print("Start at day ", self.days[self.starting_from])
        machine_type = "Truck"
        print("For machine type: ", machine_type)

        df_training_all = pd.DataFrame()
        df_testing_all = pd.DataFrame()

        for day in tqdm(
            self.days[self.starting_from : self.starting_from + self.nb_days]
        ):
            trip = dataloader.TripsLoader(day)
            for _, machine in trip._machines.items():
                if machine.machine_type == machine_type:
                    automated_for_given_machine = PrepareMachineData(machine)
                    automated_for_given_machine.get_speed_and_acceleration()
                    df_vehicle = automated_for_given_machine.construct_df_for_training(
                        self.group_size
                    )

                    X, y = (
                        df_vehicle.drop(["output_labels"], axis=1),
                        df_vehicle["output_labels"],
                    )
                    # each vehicle should be represented 20% for each day in the test data
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

            Path(self.work_dir).mkdir(parents=True, exist_ok=True)
            df_training_all.to_csv(
                f"{self.work_dir}/{self.training_data_name}_{self.nb_days}_days.csv",
                sep=",",
                index=False,
            )
            df_testing_all.to_csv(
                f"{self.work_dir}/{self.test_data_name}_{self.nb_days}_days.csv",
                sep=",",
                index=False,
            )

    def fit(self, stopping_rounds: int = 2):
        df_training = pd.read_csv(
            f"{self.work_dir}/{self.training_data_name}_{self.nb_days}_days.csv"
        )

        X_train, X_val, y_train, y_val = split_data_into_training_and_validation(
            df_training
        )

        self.booster_record_eval = {}
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
            eval_metric=self.LightGBMParams["metric"],
            eval_names=["Train", "Val"],
            callbacks=[
                early_stopping(stopping_rounds=stopping_rounds),
                record_evaluation(self.booster_record_eval),
            ],
        )

        # Save training time and validaton data error at termination
        with open(f"{self.work_dir}/track_performance.txt", "a") as f:
            f.write(
                f"...\nTraining time: {time.perf_counter() - t0} s\nData set: {self.training_data_name}_{self.nb_days}_days.\nValidation multi logloss: {self.booster_record_eval['Val']['multi_logloss'][-1]}\n"
            )
            f.flush()

        joblib.dump(model, f"{self.work_dir}/lgm_model_{self.nb_days}_days.bin")

    def plot_feature_importances(self):
        # Save feature importances as png
        fig_fi = lgbm.plot_importance(
            joblib.load(f"{self.work_dir}/lgm_model_{self.nb_days}_days.bin"),
            figsize=(15, 10),
        ).figure
        fig_fi.tight_layout()
        fig_fi.savefig(f"{self.work_dir}/feature_importance.png")
        plt.close(fig_fi)

    def plot_learning_curve(self):
        # Save learning curve as png
        fig_lc, ax_lc = plt.subplots(figsize=(8, 4))
        ax_lc.set_yscale("log")
        get_learning_curve(
            booster=self.booster_record_eval,
            metric=self.LightGBMParams["metric"],
            ax=ax_lc,
            dataset=f"{self.training_data_name}_{self.nb_days}_days",
        )
        fig_lc.tight_layout()
        fig_lc.savefig(f"{self.work_dir}/learning_curve.png")
        plt.close(fig_lc)

    def predict(self):
        df_testing = pd.read_csv(
            f"{self.work_dir}/{self.test_data_name}_{self.nb_days}_days.csv"
        )

        loaded_model = joblib.load(f"{self.work_dir}/lgm_model_{self.nb_days}_days.bin")

        # this is the order of the output matrix
        driving_label, dump_label, load_label = loaded_model.classes_

        pred_testing_proba: np.ndarray = loaded_model.predict_proba(
            df_testing.drop(["output_labels", "MachineID", "DateTime"], axis=1)
        )
        pred_testing_label: np.ndarray = loaded_model.predict(
            df_testing.drop(["output_labels", "MachineID", "DateTime"], axis=1)
        )

        df_testing[f"proba_{driving_label}"] = pred_testing_proba[:, 0]
        df_testing[f"proba_{dump_label}"] = pred_testing_proba[:, 1]
        df_testing[f"proba_{load_label}"] = pred_testing_proba[:, 2]
        df_testing["predicted_class"] = pred_testing_label
        df_testing.to_csv(f"{self.work_dir}/pred_test.csv", sep=",", index=False)

        # # store load and dump avg. probability
        # load_proba, dump_proba, driving_proba = get_avg_probabilities(df_testing)
        # # save to file
        # write_proba_score_test_data(load_proba, dump_proba, driving_proba)

    def results(self):
        pred_dict = {}
        df_preds = pd.read_csv(
            f"{self.work_dir}/pred_test.csv",
            sep=",",
            usecols=[
                "output_labels",
                "proba_Driving",
                "proba_Dump",
                "proba_Load",
                "predicted_class",
            ],
        )
        events = ["Load", "Driving", "Dump"]
        pred_events = ["proba_Load", "proba_Driving", "proba_Dump"]
        for event in events:
            for pred_event in pred_events:
                filtered_df = df_preds[df_preds["output_labels"] == event]
                pred_dict[f"{pred_event} | {event} "] = filtered_df[pred_event].mean()

        pd.DataFrame(
            {
                "Condition": list(pred_dict.keys()),
                "Probabilities": list(pred_dict.values()),
            }
        ).to_csv(
            f"{self.work_dir}/probabilities_{self.nb_days}_days.csv",
            index=False,
            sep=",",
        )

        y_true = df_preds["output_labels"]
        y_pred = df_preds["predicted_class"]

        # Assuming you have 'y_true' (true labels) and 'y_pred' (predicted labels) defined
        class_report = classification_report(y_true, y_pred, output_dict=True)
        self.statistics = {
            "Driving": {"accuracy": [], "precision": [], "f1-score": []},
            "Dump": {"accuracy": [], "precision": [], "f1-score": []},
            "Load": {"accuracy": [], "precision": [], "f1-score": []},
        }

        for activity in ["Driving", "Dump", "Load"]:
            self.statistics[activity]["precision"].append(
                class_report[activity]["precision"]
            )
            self.statistics[activity]["f1-score"].append(
                class_report[activity]["f1-score"]
            )

    def confusion_matrix(self):
        df_preds = pd.read_csv(
            f"{self.work_dir}/pred_test.csv",
            sep=",",
            usecols=["output_labels", "predicted_class"],
        )
        y_true = df_preds["output_labels"]
        y_pred = df_preds["predicted_class"]
        # Assuming you have 'y_true' (true labels) and 'y_pred' (predicted labels) defined
        conf_matrix = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Driving", "Dump", "Load"],
            yticklabels=["Driving", "Dump", "Load"],
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(f"{self.work_dir}/confusion_matrix.png")
        plt.show()

    def confusion_matrix_with_probabilities(self):
        df_preds = pd.read_csv(
            f"{self.work_dir}/pred_test.csv",
            sep=",",
            usecols=[
                "output_labels",
                "predicted_class",
                "proba_Driving",
                "proba_Dump",
                "proba_Load",
            ],
        )

        labels = ["Driving", "Dump", "Load"]
        avg_probs = np.zeros((len(labels), len(labels)))

        for i, true_label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                mask = df_preds["output_labels"] == true_label
                avg_prob = df_preds.loc[mask, f"proba_{pred_label}"].mean()
                avg_probs[i, j] = avg_prob

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            avg_probs,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )

        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Average Predicted Probabilities")
        plt.tight_layout()
        plt.savefig(f"{self.work_dir}/average_predicted_probabilities.png")
        plt.show()


# %%

if __name__ == "__main__":
    myModel = LoadDumpLightGBM(nb_days=30, group_size=10)
    myModel.load_data()
    myModel.fit()
    myModel.predict()
    myModel.results()
    myModel.confusion_matrix()
# %%
