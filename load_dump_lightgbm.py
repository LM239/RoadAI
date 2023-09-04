# %%
import dataloader
from typing import Literal
from pydantic import BaseModel
from tqdm import tqdm
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px
import ipyleaflet as L
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from schemas import Machine, Position
import geopy.distance
import matplotlib.pyplot as plt
import joblib
import lightgbm as lgbm
import time
from lightgbm import early_stopping, record_evaluation, LGBMModel
from sklearn.metrics import classification_report
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# %%


class points_times(BaseModel):
    points: list[tuple[float, float]] = []
    times: list[datetime] = []


class stats(BaseModel):  # Represents actual data
    all_positions: list[Position] = []  # Positions recorded during a day
    load: points_times = points_times()  # Load points and times
    dump: points_times = points_times()  # Dump points and times
    day_speeds: list[float] = []        # Speeds
    day_acceleration: list[float] = []
    day_dists: list[float] = []         # Distances between each recording
    day_times: list[datetime] = []      # Timestamp for two above lists
    # Inner product of consecutive normalized driving vectors
    inner_prods: list[float] = []
    # lat1_minus_lat0 = []
    # lon1_minus_lon0 = []


class automated_load_dump_for_machine():

    def __init__(self,
                 machine_data: Machine):
        #  load_cluster_centers: typing.Any,# Both should be list[tuple[float, float]], should implement
        #  dump_cluster_centers: typing.Any) -> None:

        self.machine = machine_data
        self.stats = stats()

        all_pos = [trips.positions for trips in self.machine.trips]
        self.stats.all_positions = [
            item for sublist in all_pos for item in sublist]
        self.stats.load.points = [
            trips.load_latlon for trips in self.machine.trips]
        self.stats.load.times = [
            trips.positions[0].timestamp for trips in self.machine.trips]
        self.stats.dump.points = [
            trips.dump_latlon for trips in self.machine.trips]

        actual_dump_times = []
        for t in self.machine.trips:  # Not pretty, because we don't have dump time in trip info by default
            temp_dump_laton = t.dump_latlon  # Must match latlons
            for position in t.positions:
                if temp_dump_laton == (position.lat, position.lon):
                    actual_dump_times.append(position.timestamp)
                    break
        self.stats.dump.times = actual_dump_times

        # self.meters_from_last_act = []
        # self.seconds_since_last_act = []
        # self.is_next_load = []

    def get_data(self):

        # We keep track of how many meters we have driven since last dump or load
        meters_since_last_activity = 0
        time_since_last_activity = 0
        self.stats.day_times.append(self.stats.all_positions[0].timestamp)
        # speed is added in the loop

        # We start predicting. Are going to iterate over all positions, from first to last
        for i in range(1, len(self.stats.all_positions)-1):

            next_pos = self.stats.all_positions[i+1]
            current_pos = self.stats.all_positions[i]
            prev_pos = self.stats.all_positions[i-1]

            # Meters driven since last timestamp
            meters_driven = geopy.distance.geodesic(
                (current_pos.lat, current_pos.lon), (prev_pos.lat, prev_pos.lon)).m

            meters_since_last_activity += meters_driven

            # Meters driven since last timestamp

            # this is the speed at point i-1 (forward derivative)
            # Seconds passed since last timestamp
            seconds_gone_i_minus_1 = (
                current_pos.timestamp-prev_pos.timestamp).total_seconds()
            time_since_last_activity += seconds_gone_i_minus_1

            seconds_gone_i = (next_pos.timestamp -
                              current_pos.timestamp).total_seconds()
            meters_driven_i = geopy.distance.geodesic(
                (next_pos.lat, next_pos.lon), (current_pos.lat, current_pos.lon)).m
            # if time duplicates, use a speed equal NaN
            try:
                speed_ms_i_minus_1 = meters_driven/seconds_gone_i_minus_1  # m/s
                speed_ms_i = meters_driven_i/seconds_gone_i  # m/s

                # speed_kmh_i_minus_1 = speed_ms_i_minus_1*3.6  # km/h
                # speed_kmh_i = (speed_ms_i)*3.6                # km/h
                acceleration_i_minus_1 = (
                    speed_ms_i - speed_ms_i_minus_1) / (seconds_gone_i_minus_1)  # m/s^2
            except ZeroDivisionError:
                # speed_kmh_i_minus_1 = np.nan
                speed_ms_i_minus_1 = np.nan
                acceleration_i_minus_1 = np.nan

            self.stats.day_acceleration.append(acceleration_i_minus_1)  # m/s^2
            self.stats.day_speeds.append(speed_ms_i_minus_1)           # m/s

            self.stats.day_times.append(current_pos.timestamp)

            # if we have either load or dump, distance and time from last activity is set to 0
            for sublist in [self.stats.load.points, self.stats.dump.points]:
                # , self.stats.dump.points]:
                if (self.stats.all_positions[i].lat, self.stats.all_positions[i].lon) in sublist:
                    meters_since_last_activity = 0
                    time_since_last_activity = 0

    def get_df_with_ml_data(self):
        load_times_set = set(self.stats.load.times)
        dump_times_set = set(self.stats.dump.times)
        positions = self.stats.all_positions
        latitude = [sublist.lat for sublist in positions]
        longitude = [sublist.lon for sublist in positions]
        uncertainty = [sublist.uncertainty for sublist in positions]
        lat1_minus_lat0 = [latitude[i] - latitude[i-1]
                           for i in range(1, len(latitude))]
        lon1_minus_lon0 = [longitude[i] - longitude[i-1]
                           for i in range(1, len(longitude))]
        # append some value to be removed after df is constructed
        lat1_minus_lat0.append(lat1_minus_lat0[-1])
        lon1_minus_lon0.append(lon1_minus_lon0[-1])
        speed_north_south = np.zeros_like(np.array(latitude))
        speed_east_west = np.zeros_like(np.array(latitude))

        # add some speed to the day_speeds list, as we we dont have the speed of the last data point ( see for loop)
        # we have to add a value as the speed in the last point is not defined as it uses forward derivative
        for _ in range(2):
            self.stats.day_speeds.append(np.nan)
            self.stats.day_acceleration.append(np.nan)
        for idx in range(1, len(latitude)-1):
            try:
                total_seconds = (
                    self.stats.day_times[idx] - self.stats.day_times[idx-1]).total_seconds()
                speed_east_west[idx-1] = ((longitude[idx] -
                                          longitude[idx-1]) / total_seconds)
                speed_north_south[idx-1] = ((latitude[idx] -
                                            latitude[idx-1]) / total_seconds)
            except ZeroDivisionError:
                speed_east_west[idx-1] = np.nan
                speed_north_south[idx-1] = np.nan

        # add another day time as the for loop excludes the last value
        # this value will be removed anyways
        self.stats.day_times.append(self.stats.day_times[-1])
        load = [time in load_times_set for time in self.stats.day_times]
        dump = [time in dump_times_set for time in self.stats.day_times]
        # return True if either dump or load is True
        output_labels = [d or l for d, l in zip(dump, load)]

        for i in range(len(output_labels)):
            current_time = self.stats.day_times[i]  # The current timestamp
            if current_time in self.stats.load.times:
                output_labels[i] = "Load"
            elif current_time in self.stats.dump.times:
                output_labels[i] = "Dump"
            else:
                output_labels[i] = "Driving"

        df = pd.DataFrame({
            "MachineID": [self.machine.machine_id]*len(self.stats.day_times),
            "DateTime": self.stats.day_times,
            # "Time_from_start": [(time.min - self.stats.day_times[0].min) for time in self.stats.day_times],
            "Speed": self.stats.day_speeds,
            "Acceleration": self.stats.day_acceleration,
            # "Inner_products": self.stats.inner_prods,
            "Latitude": latitude,
            "Longitude": longitude,
            "Uncertainty": uncertainty,
            "Lat1_minus_lat0": lat1_minus_lat0,
            "Lon1_minus_lon0": lon1_minus_lon0,
            "speed_north_south": speed_north_south,
            "speed_east_west": speed_east_west,
            # "km_from_last_event": self.meters_from_last_act,
            # "seconds_from_last_event": self.seconds_since_last_act,
            # "is_next_load": self.is_next_load,
            "output_labels": output_labels
        })

        # filter df to remove the rows after the last dump
        last_row = df.query('output_labels == "Dump"').index[-1]

        df = df.loc[:last_row]

        # Here we could try to merge 5 and 5 points - example
        group_size = 5
        result_df = pd.DataFrame()

        for i in range(group_size):
            sub_df = df.iloc[i::group_size]
            sub_df = sub_df.reset_index(drop=True)
            new_column_names = {}
            for column_name in df.columns:
                new_column_names[column_name] = f'{column_name}_{i}'

            # Rename the columns with the updated names
            sub_df.rename(columns=new_column_names, inplace=True)

            if i == 0:
                result_df = sub_df
            else:
                result_df = pd.concat([result_df, sub_df], axis=1)

        def custom_function(row):
            for col in result_df.columns:
                if 'output_labels' in col:
                    if row[col] == 'Load':
                        return 'Load'
                    elif row[col] == 'Dump':
                        return 'Dump'
            return 'Driving'

        result_df['output_labels'] = result_df.apply(custom_function, axis=1)
        result_df.rename(columns={'DateTime_0': 'DateTime'}, inplace=True)
        result_df.rename(columns={'MachineID_0': 'MachineID'}, inplace=True)
        for i in range(group_size):
            result_df = result_df.drop(f'output_labels_{i}', axis=1)
            if i != 0:
                result_df = result_df.drop(f'DateTime_{i}', axis=1)
                result_df = result_df.drop(f'MachineID_{i}', axis=1)
        return result_df  # df


def load_training_csv_files(
    name_of_data_set: str,
) -> pd.DataFrame:
    """
    Loads available training data
    """
    return pd.read_csv(
        f"data/ml_model_data/training_data/{name_of_data_set}", delimiter=","
    )


def split_data_into_training_and_validation(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and testing(unseen data)
    """
    X, y = (
        df.drop(["MachineID", "output_labels", "DateTime"], axis=1),
        df["output_labels"],
    )
    # delete preds if they exist
    # _delete_pred_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40
    )
    return (X_train, X_test, y_train, y_test)


def plot_learning_curve(
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


def get_avg_probabilities(df_pred: pd.DataFrame) -> tuple[float, float, float]:
    # filter to return only rows where we have loads and dumps
    true_loads_rows = df_pred.loc[df_pred["output_labels"]
                                  == "Load", "proba_Load"]
    true_dumps_rows = df_pred.loc[df_pred["output_labels"]
                                  == "Dump", "proba_Dump"]
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


class LoadDumpLightGBM:

    def __init__(self, nb_days: int = 1,
                 starting_from: int = 0,
                 merge_timestamps: int = 5,
                 work_dir: str = 'data/ml_model_data/class_data') -> None:

        self.nb_days = nb_days
        self.starting_from = starting_from
        self.merge_timestamps = merge_timestamps  # Må sendes videre til group size
        self.work_dir = work_dir
        self.training_data_name = 'my_train_from_class'
        self.test_data_name = 'my_test_from_class'

        self.LightGBMParams = {
            "boosting_type": "gbdt",
            "metric": "multi_logloss",
            "num_leaves": 31,
            "learning_rate": 0.5,
            "feature_fraction": 1,
            "num_boost_round": 1000,
        }

    def load_data(self):

        days = [csv_file.split(".csv")[0]
                for csv_file in os.listdir("data/GPSData/trips")]
        machine_type = "Truck"

        df_training_all = pd.DataFrame()
        df_testing_all = pd.DataFrame()

        for day in tqdm(days[self.starting_from:self.starting_from+self.nb_days]):
            trip = dataloader.TripsLoader(day)
            for unique_vehicle in trip._machines.keys():
                temp_machine = trip._machines[unique_vehicle]
                if temp_machine.machine_type == machine_type:
                    # machine_of_interest = trip._machines[unique_vehicle]
                    automated_for_given_machine = automated_load_dump_for_machine(
                        temp_machine)
                    automated_for_given_machine.get_data()
                    df_vehicle = automated_for_given_machine.get_df_with_ml_data()

                    X, y = df_vehicle.drop(
                        ["output_labels"], axis=1), df_vehicle["output_labels"]
                    # each vehicle should be represented 20% for each day in the test data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=40)

                    df_training = pd.concat(
                        [X_train, y_train], axis=1).sort_values(by="DateTime")
                    # grouped = df_training.groupby(df_training.index //3)
                    # df_training.drop("DateTime")
                    # df_training.insert(1,"DateTime start": d)

                    df_training_all = pd.concat(
                        [df_training_all, df_training], axis=0)
                    # add the training and testing data to the total dataframe by row
                    df_testing = pd.concat(
                        [X_test, y_test], axis=1).sort_values(by="DateTime")
                    df_testing_all = pd.concat(
                        [df_testing_all, df_testing], axis=0)

            df_training_all.dropna(inplace=True)
            df_testing_all.dropna(inplace=True)

            df_training_all.to_csv(
                f"{self.work_dir}/{self.training_data_name}.csv", sep=',', index=False)
            df_testing_all.to_csv(
                f"{self.work_dir}/{self.test_data_name}.csv", sep=',', index=False)

    def fit(self):
        df_training = pd.read_csv(
            f"{self.work_dir}/{self.training_data_name}.csv")

        # split again to get data to validate against during iterations
        X_train, X_val, y_train, y_val = split_data_into_training_and_validation(
            df_training
        )
        # predict both load and dump and save the models
        fig_lc, ax_lc = plt.subplots(figsize=(8, 4))
        ax_lc.set_yscale("log")

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
            eval_metric=self.LightGBMParams["metric"],
            eval_names=["Train", "Val"],
            callbacks=[
                early_stopping(stopping_rounds=2),
                record_evaluation(booster_record_eval),
            ],
        )

        plot_learning_curve(
            booster=booster_record_eval,
            metric=self.LightGBMParams["metric"],
            ax=ax_lc,
            dataset=self.training_data_name,
        )
        fig_lc.tight_layout()
        fig_lc.savefig(f"{self.work_dir}/learning_curve.png")

        # plot feature importances
        fig_fi = lgbm.plot_importance(model).figure
        fig_fi.tight_layout()
        fig_fi.savefig(f"{self.work_dir}/feature_importance.png")

        # save training time, val_error at termination
        with open(f"{self.work_dir}/track_performance.txt", "a") as f:
            f.write(
                f"...\nTraining time: {time.perf_counter() - t0} s\nData set: {self.training_data_name}.\nValidation multi logloss: {booster_record_eval['Val']['multi_logloss'][-1]}\n"
            )

        joblib.dump(model, f"{self.work_dir}/lgm_model.bin")

    def predict(self):
        """
        Load model and predict on unseen data
        """
        df_testing = pd.read_csv(f"{self.work_dir}/{self.test_data_name}.csv")

        loaded_model = joblib.load(f"{self.work_dir}/lgm_model.bin")

        # this is the order of the output matrix
        driving_label, dump_label, load_label = loaded_model.classes_

        # pred_training = loaded_model.predict_proba(X_train)
        pred_testing: np.ndarray = loaded_model.predict_proba(
            df_testing.drop(["output_labels", "MachineID", "DateTime"], axis=1)
        )
        pred_class_testing: np.ndarray = loaded_model.predict(
            df_testing.drop(["output_labels", "MachineID", "DateTime"], axis=1)
        )

        df_testing[f"proba_{driving_label}"] = pred_testing[:, 0]
        df_testing[f"proba_{dump_label}"] = pred_testing[:, 1]
        df_testing[f"proba_{load_label}"] = pred_testing[:, 2]
        df_testing["predicted_class"] = pred_class_testing
        df_testing.to_csv(f"{self.work_dir}/pred_test.csv",
                          sep=",", index=False)

        ############ the same piece of info can be found in preds/probabilities #############
        # store load and dump avg. probability
        load_proba, dump_proba, driving_proba = get_avg_probabilities(
            df_testing)
        # save to file
        write_proba_score_test_data(load_proba, dump_proba, driving_proba)

    def results(self):
        pred_dict = {}
        df_preds = pd.read_csv(
            f"{self.work_dir}/pred_test.csv",
            sep=",",
            usecols=["output_labels", "proba_Driving",
                     "proba_Dump", "proba_Load", "predicted_class"],
        )
        events = ["Load", "Driving", "Dump"]
        pred_events = ["proba_Load", "proba_Driving", "proba_Dump"]
        for event in events:
            for pred_event in pred_events:
                filtered_df = df_preds[df_preds["output_labels"] == event]
                pred_dict[f"{pred_event} | {event} "] = filtered_df[pred_event].mean()

        pd.DataFrame(
            {"Condition": list(pred_dict.keys()),
             "Probabilities": list(pred_dict.values())}
        ).to_csv(f"{self.work_dir}/probabilities{self.nb_days}.csv", index=False, sep=",")

        y_true = df_preds["output_labels"]
        y_pred = df_preds["predicted_class"]

        # Assuming you have 'y_true' (true labels) and 'y_pred' (predicted labels) defined
        class_report = classification_report(y_true, y_pred, output_dict=True)
        statistics = {"Driving": {"accuracy": [], "precision": [], "f1-score": []},
                      "Dump": {"accuracy": [], "precision": [], "f1-score": []},
                      "Load": {"accuracy": [], "precision": [], "f1-score": []}}

        for activity in ["Driving", "Dump", "Load"]:
            statistics[activity]["precision"].append(
                class_report[activity]["precision"])
            statistics[activity]["f1-score"].append(
                class_report[activity]["f1-score"])

        with open("person.txt", "w") as fp:
            json.dump(statistics, fp)  # encode dict into JSON

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
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[
                    'Driving', 'Dump', 'Load'], yticklabels=['Driving', 'Dump', 'Load'])
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

# %%


myModel = LoadDumpLightGBM(nb_days=3)
myModel.load_data()
myModel.fit()
myModel.predict()
myModel.results()
myModel.confusion_matrix()
# %%
