import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataloader import TripsLoader
from pydantic import BaseModel
import numpy as np
import geopy.distance
from datetime import datetime
import pandas as pd
from schemas import Machine, Position


class points_times(BaseModel):
    points: list[tuple[float, float]] = []
    times: list[datetime] = []


class predicted_load_dump(BaseModel):
    load: points_times = points_times()
    dump: points_times = points_times()


class stats(BaseModel):  # Represents actual data
    all_positions: list[Position] = []  # Positions recorded during a day
    load: points_times = points_times()  # Load points and times
    dump: points_times = points_times()  # Dump points and times
    day_speeds: list[float] = []  # Speeds
    day_acceleration: list[float] = []
    day_dists: list[float] = []  # Distances between each recording
    day_times: list[datetime] = []  # Timestamp for two above lists
    inner_prods: list[
        float
    ] = []  # Inner product of consecutive normalized driving vectors


class GetDataForModel:
    def __init__(self, machine_data: Machine):
        self.machine = machine_data
        self.predicted = predicted_load_dump()
        self.stats = stats()

        all_pos = [trips.positions for trips in self.machine.trips]
        self.stats.all_positions = [item for sublist in all_pos for item in sublist]
        self.stats.load.points = [trips.load_latlon for trips in self.machine.trips]
        self.stats.load.times = [
            trips.positions[0].timestamp for trips in self.machine.trips
        ]
        self.stats.dump.points = [trips.dump_latlon for trips in self.machine.trips]

        actual_dump_times = []
        for (
            t
        ) in (
            self.machine.trips
        ):  # Not pretty, because we don't have dump time in trip info by default
            temp_dump_laton = t.dump_latlon  # Must match latlons
            for position in t.positions:
                if temp_dump_laton == (position.lat, position.lon):
                    actual_dump_times.append(position.timestamp)
                    break
        self.stats.dump.times = actual_dump_times

    def get_data(self):
        # We keep track of how many meters we have driven since last dump or load
        meters_since_last_activity = 0
        time_since_last_activity = 0
        self.stats.day_times.append(self.stats.all_positions[0].timestamp)

        for i in range(1, len(self.stats.all_positions) - 1):
            next_pos = self.stats.all_positions[i + 1]
            current_pos = self.stats.all_positions[i]
            prev_pos = self.stats.all_positions[i - 1]

            # Meters driven since last timestamp
            meters_driven = geopy.distance.geodesic(
                (current_pos.lat, current_pos.lon), (prev_pos.lat, prev_pos.lon)
            ).m

            meters_since_last_activity += meters_driven

            # Meters driven since last timestamp

            # this is the speed at point i-1 (forward derivative)
            # Seconds passed since last timestamp
            seconds_gone_i_minus_1 = (
                current_pos.timestamp.to_pydatetime()
                - prev_pos.timestamp.to_pydatetime()
            ).total_seconds()
            time_since_last_activity += seconds_gone_i_minus_1

            seconds_gone_i = (
                next_pos.timestamp.to_pydatetime()
                - current_pos.timestamp.to_pydatetime()
            ).total_seconds()
            meters_driven_i = geopy.distance.geodesic(
                (next_pos.lat, next_pos.lon), (current_pos.lat, current_pos.lon)
            ).m
            # if time duplicates, use a speed equal NaN
            try:
                speed_ms_i_minus_1 = meters_driven / seconds_gone_i_minus_1  # m/s
                speed_ms_i = meters_driven_i / seconds_gone_i  # m/s

                acceleration_i_minus_1 = (speed_ms_i - speed_ms_i_minus_1) / (
                    seconds_gone_i_minus_1
                )  # m/s^2
            except ZeroDivisionError:
                # speed_kmh_i_minus_1 = np.nan
                speed_ms_i_minus_1 = np.nan
                acceleration_i_minus_1 = np.nan

            self.stats.day_acceleration.append(acceleration_i_minus_1)  # m/s^2
            self.stats.day_speeds.append(speed_ms_i_minus_1)  # m/s

            self.stats.day_times.append(current_pos.timestamp)

            # if we have either load or dump, distance and time from last activity is set to 0
            for sublist in [self.stats.load.points, self.stats.dump.points]:
                if (
                    self.stats.all_positions[i].lat,
                    self.stats.all_positions[i].lon,
                ) in sublist:  # , self.stats.dump.points]:
                    meters_since_last_activity = 0
                    time_since_last_activity = 0

    def get_df_with_ml_data(self):
        load_times_set = set(self.stats.load.times)
        dump_times_set = set(self.stats.dump.times)
        positions = self.stats.all_positions
        latitude = [sublist.lat for sublist in positions]
        longitude = [sublist.lon for sublist in positions]
        uncertainty = [sublist.uncertainty for sublist in positions]
        lat1_minus_lat0 = [
            latitude[i] - latitude[i - 1] for i in range(1, len(latitude))
        ]
        lon1_minus_lon0 = [
            longitude[i] - longitude[i - 1] for i in range(1, len(longitude))
        ]
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
        for idx in range(1, len(latitude) - 1):
            try:
                total_seconds = (
                    self.stats.day_times[idx].to_pydatetime()
                    - self.stats.day_times[idx - 1].to_pydatetime()
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

        return df


class DataLoader:
    def __init__(
        self,
        machine_type: str = "Truck",
        n_days: int = 20,
        gps_data_path: str = "data/GPSData/trips",
    ):
        self.machine_type = machine_type
        self.n_days = n_days
        self.days = [
            csv_file.split(".csv")[0] for csv_file in os.listdir(gps_data_path)
        ]
        self.df_training_all = pd.DataFrame()
        self.df_testing_all = pd.DataFrame()

    def create_training_and_testing_datasets(self):
        for day in self.days[: self.n_days]:
            trip = TripsLoader(day)
            for unique_vehicle in trip._machines.keys():
                temp_machine = trip._machines[unique_vehicle]
                if temp_machine.machine_type == self.machine_type:
                    automated_for_given_machine = GetDataForModel(temp_machine)
                    automated_for_given_machine.get_data()
                    df_vehicle = automated_for_given_machine.get_df_with_ml_data()

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

                    self.df_training_all = pd.concat(
                        [self.df_training_all, df_training], axis=0
                    )

                    # add the training and testing data to the total dataframe by row
                    df_testing = pd.concat([X_test, y_test], axis=1).sort_values(
                        by="DateTime"
                    )
                    self.df_testing_all = pd.concat(
                        [self.df_testing_all, df_testing], axis=0
                    )


class LoadDumpPredictor:
    def __init__(
        self,
        machine_type: str = "Truck",
        n_days: int = 20,
        gps_data_path: str = "data/GPSData/trips",
    ):
        self.machine_type = machine_type
        self.n_days = n_days
        self.gps_data_path = gps_data_path
        self.df_training_all = pd.DataFrame()
        self.df_testing_all = pd.DataFrame()
        self.days = [
            csv_file.split(".csv")[0] for csv_file in os.listdir(gps_data_path)
        ]
        self.model = None

    def load_and_prepare_data(self):
        print("something")
        dataloader = DataLoader(self.machine_type, self.n_days, self.gps_data_path)
        dataloader.create_training_and_testing_datasets()
        print("Data successfully loaded and converted to necessary structure")
        self.df_testing_all = dataloader.df_testing_all
        self.df_training_all = dataloader.df_training_all

    def train_lightgbm_and_plot_metric(self):
        # Your LightGBM training and metric plotting logic here
        pass

    def load_and_predict(self):
        # Your prediction logic here
        pass

    def save_dfs_with_ml_data(self):
        # Your existing `save_dfs_with_ml_data` logic here
        pass

    def run_pipeline(self):
        self.load_and_prepare_data()
        self.save_dfs_with_ml_data()
        self.train_lightgbm_and_plot_metric()
        self.load_and_predict()
