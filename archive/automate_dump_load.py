# %%
import helper_functions.dataloader as dataloader
from pydantic import BaseModel
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import geopy.distance
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from tqdm import tqdm
import pandas as pd
from helper_functions.schemas import Position
import ipyleaflet as L


# Helper function to check if timestamp is close to timestamps within a list
def within_activity_seconds(target_timestamp, timestamp_list, time_threshold):
    for ts in timestamp_list:
        time_difference = abs((ts - target_timestamp).total_seconds())

        if time_difference <= time_threshold:  # 3 minutes in seconds
            return True

    return False


class Points_times(BaseModel):
    points: list[tuple[float, float]] = []  # Latlons
    times: list[datetime] = []


class Criteria(BaseModel):
    ###############################################################
    ###########Criterias for prediction of load and dump###########
    ###############################################################

    optimal_K: int = 50  # Nb of clusters for work areas

    meters_from_area: int = 30  # Radius from a cluster center

    seconds_for_vector: int = (
        10  # "Length" of vector used to determine if vehicle is reversing
    )

    speed_limit: int = 30  # Cannot be loading or dumping if speed higher than this. Don't like such a high number

    meters_since_last_activity: int = 300  # Meters driven since last load/dump

    minutes_load: int = (
        3  # Look at distance driven last x minutes before a possible load
    )

    max_sum_last_x_minutes_load: int = (
        1000  # Max meters driven during the last x minutes
    )

    minutes_dump: int = (
        3  # Look at distance driven last x minutes before a possible load
    )

    max_sum_last_x_minutes_dump: int = (
        1000  # Max meters driven during the last x minutes
    )

    inner_prod_threshold: float = (
        0.80  # A threshold to pick up possible reversal movement from dot product
    )

    ###############################################################
    ################Criterias for idle computation#################
    ###############################################################

    avg_speed_limit: int = (
        5  # Average speed max for a given time, before we decide its idle time
    )
    time_for_avg_speed: int = 30  # Seconds for avg speed before we decide it is idle time. Can reduce sensitivity

    min_idle_period: int = (
        30  # Minimum number of seconds before it can become idle period
    )

    dump_expected_time: int = 60  # Seconds expected for dump time

    load_expected_time: int = 300  # Seconds expected for load time

    idle_speed_limit: int = (
        10  # Maximum speed km/h to say vehicle stands still, in relation to idle time
    )


class Predicted_load_dump(BaseModel):
    load: Points_times = Points_times()
    dump: Points_times = Points_times()


class Stats(BaseModel):  # Represents actual data
    all_positions: list[Position] = []  # Positions recorded during a day
    load: Points_times = Points_times()  # Load points and times
    dump: Points_times = Points_times()  # Dump points and times
    day_stats_set: bool = False  # If the three day statistics have been set
    day_speeds: list[float] = []  # Speeds
    day_dists: list[float] = []  # Distances between each recording
    day_times: list[datetime] = []  # Timestamp for two above lists
    inner_prods: list[float] = []  # Inner product of consecutive normalized vectors
    list_of_idle_times: list[Points_times] = []  # List of all times idle during a day


class Automated_load_dump_for_machine:
    """Compute load, dump and idle time for selected machine on given day

    ...

    Attributes
    ----------
    machine : Machine
    predicted : Predicted_load_dump
        Class keeping track of predicted load and dump points(latlons) and times
    stats : Stats
        Class with actual and computed values
    criterias: Criteria
        Class with parameters for algorithms computing load/dump/idle time
    load_cluster_centers: List of (lat,lons)
        Used to determine if machine is in proximity to normal load area
    dump_cluster_centers: List of (lat,lons)
        Used to determine if machine is in proximity to normal dump area

    Methods
    -------
    generate_load_dump_clusters(day: str):
        Generates cluster centers, usually want day before
    predict_loaddump():
        Predicts/computes load and dump points and times
    prediction_time_plot():
        Time plot of actual versus predicted load and dump
    prediction_gantt_plot():
        Gantt plot of actual and predicted trips
    find_idle_time():
        Computes times that are idle during day
    idle_report():
        Short summary of idle statistics, and heatmap of positions
    idle_time_plot():
        Time plot of where machine was idle, similar to prediction_time_plot
    """

    def __init__(self, day: str, machine_nb: int) -> None:
        # Loading gps data for selected day and day before
        print("Loading data for day...")
        trip = dataloader.TripsLoader(day)

        self.machine = trip._machines[machine_nb]
        self.predicted = Predicted_load_dump()
        self.stats = Stats()
        self.criterias = Criteria()

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
        print("Generating clusters for load and dump...")
        (
            self.load_cluster_centers,
            self.dump_cluster_centers,
        ) = self.generate_load_dump_clusters(day)

        # These four lines should be rewritten, to a class or something
        self.entering_load_working_area = []
        self.exiting_load_working_area = []
        self.entering_dump_working_area = []
        self.exiting_dump_working_area = []

    def generate_load_dump_clusters(self, day: str):
        # Use previous day data to create clustering
        # Convert the date string to a datetime object
        date_obj = datetime.strptime(day, "%m-%d-%Y")
        # Subtract one day using timedelta
        new_date = date_obj - timedelta(days=1)
        # Format the new date back to the desired format
        day_before = new_date.strftime("%m-%d-%Y")

        trip_day_before = dataloader.TripsLoader(day_before)
        all_load_positions_for_day_before = []
        all_dump_positions_for_day_before = []
        for machine_number in trip_day_before._machines.keys():
            temp_machine = trip_day_before._machines[machine_number]
            if temp_machine.machine_type == self.machine.machine_type:
                all_load_positions_for_day_before.append(
                    [trip.load_latlon for trip in temp_machine.trips]
                )
                all_dump_positions_for_day_before.append(
                    [trip.dump_latlon for trip in temp_machine.trips]
                )

        all_load_positions_for_day_before = [
            item for sublist in all_load_positions_for_day_before for item in sublist
        ]
        all_dump_positions_for_day_before = [
            item for sublist in all_dump_positions_for_day_before for item in sublist
        ]

        # Assuming 'coordinates' is list of tuples [(lat1, lon1), (lat2, lon2), ...]
        load_coordinates_array = np.array(all_load_positions_for_day_before)
        dump_coordinates_array = np.array(all_dump_positions_for_day_before)

        # Fit the KMeans model with the optimal K value
        load_kmeans = KMeans(
            n_clusters=self.criterias.optimal_K, random_state=42, n_init="auto"
        )
        dump_kmeans = KMeans(
            n_clusters=self.criterias.optimal_K, random_state=42, n_init="auto"
        )
        load_kmeans.fit(load_coordinates_array)
        dump_kmeans.fit(dump_coordinates_array)

        # Get the coordinates of the cluster centers for the optimal K value
        load_cluster_centers = load_kmeans.cluster_centers_
        dump_cluster_centers = dump_kmeans.cluster_centers_

        return load_cluster_centers, dump_cluster_centers

    def predict_loaddump(self):
        print("Starting prediction of loads and dumps...")
        # We know first loading because that is when data begins
        self.predicted.load.points.append(
            (self.stats.all_positions[0].lat, self.stats.all_positions[0].lon)
        )
        self.predicted.load.times.append(self.stats.all_positions[0].timestamp)

        # When true, we are predicting load, else dump. Next prediction will be dump, since we have load above
        predicting_load = False

        # Initialize variables that keep track of whether or not we are in a usual area for loading or dumping
        # As given by load and dump clusters and criterias.meters_from_area
        in_load_working_area = (
            False  # Probably true, since first position is when loading
        )
        in_dump_working_area = (
            False  # Maybe false, since first position is when loading
        )

        # We determine the true value of the two above variables
        for (
            coord
        ) in (
            self.load_cluster_centers
        ):  # But verify with created clusters of load points from day before
            if (
                geopy.distance.geodesic(
                    coord,
                    (
                        self.stats.all_positions[0].lat,
                        self.stats.all_positions[0].lon,
                    ),
                ).m
                < self.criterias.meters_from_area
            ):
                in_load_working_area = True

        for (
            coord
        ) in (
            self.dump_cluster_centers
        ):  # But verify with created clusters of load points from day before
            if (
                geopy.distance.geodesic(
                    coord,
                    (
                        self.stats.all_positions[0].lat,
                        self.stats.all_positions[0].lon,
                    ),
                ).m
                < self.criterias.meters_from_area
            ):
                in_dump_working_area = True

        # We keep track of how many meters we have driven since last dump or load
        meters_since_last_activity = 0

        # We start predicting. Are going to iterate over all positions, from first to last
        for i in tqdm(range(1, len(self.stats.all_positions[1:]))):
            current_pos = self.stats.all_positions[i]
            prev_pos = self.stats.all_positions[i - 1]

            # Seconds passed since last timestamp
            seconds_gone = (current_pos.timestamp - prev_pos.timestamp).total_seconds()

            if seconds_gone > 0:
                # Meters driven since last timestamp
                meters_driven = geopy.distance.geodesic(
                    (current_pos.lat, current_pos.lon), (prev_pos.lat, prev_pos.lon)
                ).m
                meters_since_last_activity += meters_driven

                # Meters driven since last timestamp
                speed_kmh = (meters_driven / seconds_gone) * 3.6
                if not self.stats.day_stats_set:
                    # Add the speed to a list for entire day
                    self.stats.day_speeds.append(speed_kmh)

                    # Add the distance (km) between the two timestamps
                    self.stats.day_dists.append(meters_driven / 1000)

                    # Add the timestamp for the two above values
                    self.stats.day_times.append(current_pos.timestamp)

                # Compute vectors. This is a lot of code, maybe create some function?
                # Create vector for computing reverse of vehicle
                vector_start = current_pos.timestamp - timedelta(
                    seconds=self.criterias.seconds_for_vector
                )
                index_start_vector = 0
                for j, ts in enumerate(self.stats.day_times):
                    if ts >= vector_start:
                        index_start_vector = j
                        break

                current_vector = [
                    self.stats.all_positions[i].lat
                    - self.stats.all_positions[i - 3].lat,
                    self.stats.all_positions[i].lon
                    - self.stats.all_positions[i - 3].lon,
                ]
                prev_vector = [
                    self.stats.all_positions[index_start_vector].lat
                    - self.stats.all_positions[index_start_vector - 3].lat,
                    self.stats.all_positions[index_start_vector].lon
                    - self.stats.all_positions[index_start_vector - 3].lon,
                ]

                current_vector_norm = current_vector / np.linalg.norm(current_vector)
                prev_vector_norm = prev_vector / np.linalg.norm(prev_vector)
                inner_product = np.inner(current_vector_norm, prev_vector_norm)
                self.stats.inner_prods.append(inner_product)

                # Check if we are currently in a loading or dumping working area
                # If yes, we update value
                currently_in_load_working_area = False
                for coord in self.load_cluster_centers:
                    if (
                        geopy.distance.geodesic(
                            coord, (current_pos.lat, current_pos.lon)
                        ).m
                        < self.criterias.meters_from_area
                    ):
                        currently_in_load_working_area = True

                currently_in_dump_working_area = False
                for coord in self.dump_cluster_centers:
                    if (
                        geopy.distance.geodesic(
                            coord, (current_pos.lat, current_pos.lon)
                        ).m
                        < self.criterias.meters_from_area
                    ):
                        currently_in_dump_working_area = True

                # Could be interesting to mark where we enter and exit load and dump zones. Can be done with this code.
                # Can be used for plotting, see notebooks for examples
                if not in_load_working_area and currently_in_load_working_area:
                    self.entering_load_working_area.append(current_pos.timestamp)
                    in_load_working_area = True
                elif in_load_working_area and not currently_in_load_working_area:
                    self.exiting_load_working_area.append(current_pos.timestamp)
                    in_load_working_area = False

                if not in_dump_working_area and currently_in_dump_working_area:
                    self.entering_dump_working_area.append(current_pos.timestamp)
                    in_dump_working_area = True
                elif in_dump_working_area and not currently_in_dump_working_area:
                    self.exiting_dump_working_area.append(current_pos.timestamp)
                    in_dump_working_area = False

                # Logic for predicting loading point
                if (
                    speed_kmh < self.criterias.speed_limit
                    and meters_since_last_activity
                    > self.criterias.meters_since_last_activity
                ):
                    if predicting_load:
                        if in_load_working_area:
                            last_min_start = current_pos.timestamp - timedelta(
                                minutes=self.criterias.minutes_load
                            )
                            index_start_minute = 0
                            for i, ts in enumerate(self.stats.day_times):
                                if ts >= last_min_start:
                                    index_start_minute = i
                                    break
                            sum_over_last_minute = np.sum(
                                self.stats.day_speeds[index_start_minute:]
                            )

                            if (
                                sum_over_last_minute
                                < self.criterias.max_sum_last_x_minutes_load
                            ):
                                self.predicted.load.points.append(
                                    (current_pos.lat, current_pos.lon)
                                )
                                self.predicted.load.times.append(current_pos.timestamp)

                                # Have now predicted a load, next dump
                                predicting_load = False
                                meters_since_last_activity = 0

                    else:  # Predicting dump
                        if in_dump_working_area:
                            last_min_start = current_pos.timestamp - timedelta(
                                minutes=self.criterias.minutes_dump
                            )
                            index_start_minute = 0
                            for i, ts in enumerate(self.stats.day_times):
                                if ts >= last_min_start:
                                    index_start_minute = i
                                    break
                            sum_over_last_minute = np.sum(
                                self.stats.day_speeds[index_start_minute:]
                            )

                            if (
                                sum_over_last_minute
                                < self.criterias.max_sum_last_x_minutes_dump
                                and self.stats.inner_prods[-1]
                                < self.criterias.inner_prod_threshold
                            ):  # Not ideal, want to instead pick the best among times, not just the first viable option, but difficult with live tracking.
                                self.predicted.dump.points.append(
                                    (current_pos.lat, current_pos.lon)
                                )
                                self.predicted.dump.times.append(current_pos.timestamp)

                                # Have now predicted a dump, next load
                                predicting_load = True
                                meters_since_last_activity = 0
        self.stats.day_stats_set = True
        print("Finished prediction!")

    def prediction_time_plot(self):
        # Plots, not wanted when a lot of data
        # Create subplots with 2 rows and 1 column
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        # Add the first line plot to the first subplot
        fig.add_trace(
            go.Scatter(
                x=self.stats.day_times,
                y=self.stats.day_speeds,
                mode="lines",
                name="Speed",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.stats.load.times,
                y=[0 for a in self.stats.load.times],
                mode="markers",
                marker=dict(symbol="cross", size=10, color="red"),
                name="Load actual",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.predicted.load.times,
                y=[0 for p in self.predicted.load.times],
                mode="markers",
                marker=dict(symbol="star", size=10, color="red"),
                name="Load predicted",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.stats.dump.times,
                y=[0 for a in self.stats.dump.times],
                mode="markers",
                marker=dict(symbol="cross", size=10, color="green"),
                name="Dump actual",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.predicted.dump.times,
                y=[0 for p in self.predicted.dump.times],
                mode="markers",
                marker=dict(symbol="star", size=10, color="green"),
                name="Dump predicted",
            ),
            row=1,
            col=1,
        )

        # Add the second line plot to the second subplot
        fig.add_trace(
            go.Scatter(
                x=self.stats.day_times,
                y=np.cumsum(self.stats.day_dists),
                mode="lines",
                name="Cumulative distance",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.stats.load.times,
                y=[0 for a in self.stats.load.times],
                mode="markers",
                marker=dict(symbol="cross", size=10, color="red"),
                name="Load actual",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.predicted.load.times,
                y=[0 for p in self.predicted.load.times],
                mode="markers",
                marker=dict(symbol="star", size=10, color="red"),
                name="Load predicted",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.stats.dump.times,
                y=[0 for a in self.stats.dump.times],
                mode="markers",
                marker=dict(symbol="cross", size=10, color="green"),
                name="Dump actual",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.predicted.dump.times,
                y=[0 for p in self.predicted.dump.times],
                mode="markers",
                marker=dict(symbol="star", size=10, color="green"),
                name="Dump predicted",
            ),
            row=2,
            col=1,
        )

        # Add the third line plot
        fig.add_trace(
            go.Scatter(
                x=self.stats.day_times,
                y=self.stats.inner_prods,
                mode="lines",
                name="Inner product of vectors",
            ),
            row=3,
            col=1,
        )

        fig.update_yaxes(title_text="Km/h", row=1, col=1)
        fig.update_yaxes(title_text="Km", row=2, col=1)
        fig.update_yaxes(title_text="Dot product", row=3, col=1)
        fig.update_xaxes(title_text="Timestamp", row=3, col=1)

        # Update layout settings for both subplots
        fig.update_layout(
            title=str(
                "Subplots of speeds, cumulative distance and dot product, machine_id: "
                + str(self.machine.machine_id)
            ),
            showlegend=True,
        )

        fig.show()

    def prediction_gantt_plot(self):
        all_trips_for_machine = self.machine.trips
        start_end_each_trip_dict_actual = [
            dict(
                Start=trips.start_date,
                End=trips.end_date,
                Load=trips.load,
                Dist=trips.length,
                Id=trips.trip_id,
            )
            for trips in all_trips_for_machine
        ]
        df_pyplot_actual = pd.DataFrame(start_end_each_trip_dict_actual)

        # Predicted trips
        start_end_each_trip_dict_predicted = [
            dict(
                Start=self.predicted.load.times[i],
                End=self.predicted.load.times[i + 1],
            )
            for i in range(len(self.predicted.load.times) - 1)
        ]
        df_pyplot_predicted = pd.DataFrame(start_end_each_trip_dict_predicted)
        # Create the individual plots
        fig_actual = px.timeline(
            df_pyplot_actual,
            x_start="Start",
            x_end="End",
            custom_data=["Load", "Dist", "Id"],
            title="Actual trips",
        )
        fig_actual.update_traces(
            hovertemplate="<br>".join(
                [
                    "Start: %{base}",
                    "End: %{x}",
                    "Load: %{customdata[0]}",
                    "Distance: %{customdata[1]}",
                    "ID: %{customdata[2]}",
                ]
            )
        )
        fig_actual.update_yaxes(autorange="reversed", title_text="Trip number")
        fig_actual.update_xaxes(title_text="Timestamp")

        fig_predicted = px.timeline(
            df_pyplot_predicted, x_start="Start", x_end="End", title="Predicted trips"
        )
        fig_predicted.update_traces(
            hovertemplate="<br>".join(["Start: %{base}", "End: %{x}"])
        )
        fig_predicted.update_yaxes(autorange="reversed", title_text="Trip number")
        fig_predicted.update_xaxes(title_text="Timestamp")

        # Create a subplot with two rows and one column
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Actual trips", "Predicted trips"),
        )

        # Add the individual plots to the subplot
        fig.add_trace(fig_actual.data[0], row=1, col=1)
        fig.add_trace(fig_predicted.data[0], row=2, col=1)

        # Update subplot layout
        fig.update_layout(height=600, width=800, title_text="Trips timeline")
        fig.update_xaxes(type="date")
        fig.update_yaxes(row=1, col=1, autorange="reversed")
        fig.update_yaxes(row=2, col=1, autorange="reversed")
        fig.update_yaxes(title_text="Trip nb", row=1, col=1)
        fig.update_yaxes(title_text="Trip nb", row=2, col=1)
        fig.update_xaxes(title_text="Timestamp", row=2, col=1)

        # Show the combined subplot
        fig.show()

    def find_idle_time(self):
        print("Computing idle times...")
        # A list containing idle periods before passed to object
        temp_idle_list = Points_times()
        list_of_all_times = [pos.timestamp for pos in self.stats.all_positions]
        list_of_all_pos = [(pos.lat, pos.lon) for pos in self.stats.all_positions]

        # We start processing. Are going to iterate over all positions, from first to last
        for i in tqdm(range(1, len(self.stats.all_positions[1:]))):
            current_pos = self.stats.all_positions[i]
            prev_pos = self.stats.all_positions[i - 1]
            current_time = current_pos.timestamp

            # Seconds passed since last timestamp
            seconds_gone = (current_pos.timestamp - prev_pos.timestamp).total_seconds()

            if seconds_gone > 0:
                # Meters driven since last timestamp
                meters_driven = geopy.distance.geodesic(
                    (current_pos.lat, current_pos.lon), (prev_pos.lat, prev_pos.lon)
                ).m

                # Meters driven since last timestamp
                speed_kmh = (meters_driven / seconds_gone) * 3.6

                # Compute average speed over last x seconds, see criterias
                vector_start = current_pos.timestamp - timedelta(
                    seconds=self.criterias.time_for_avg_speed
                )
                index_start_vector = len(list_of_all_times) - 1
                for j, ts in reversed(list(enumerate(list_of_all_times))):
                    if ts <= vector_start:
                        index_start_vector = j
                        break
                seconds_between = (
                    current_pos.timestamp - list_of_all_times[j]
                ).total_seconds()
                meters_between = geopy.distance.geodesic(
                    (current_pos.lat, current_pos.lon),
                    list_of_all_pos[index_start_vector],
                ).m
                current_avg_speed = (
                    meters_between / (seconds_between + 0.000001)
                ) * 3.6

                # With this if/else we accept load and dump as idle as well
                if not self.stats.day_stats_set:
                    # Add the speed to a list for entire day
                    self.stats.day_speeds.append(speed_kmh)

                    # Add the distance (km) between the two timestamps
                    self.stats.day_dists.append(meters_driven / 1000)

                    # Add the timestamp for the two above values
                    self.stats.day_times.append(current_pos.timestamp)

                if speed_kmh < self.criterias.idle_speed_limit:
                    temp_idle_list.points.append((current_pos.lat, current_pos.lon))
                    temp_idle_list.times.append(current_time)
                    if (
                        i == len(self.stats.all_positions[1:]) - 1
                    ):  # i.e. last iteration
                        self.stats.list_of_idle_times.append(temp_idle_list)
                else:
                    if len(temp_idle_list.points) > 0:
                        self.stats.list_of_idle_times.append(temp_idle_list)
                        temp_idle_list = Points_times()  # Re-initialize

                """ # First check if we are in a dumping or loading "zone"
                if within_activity_seconds(
                    current_time,
                    self.stats.load.times,
                    self.criterias.load_expected_time,
                ) or within_activity_seconds(
                    current_time,
                    self.stats.dump.times,
                    self.criterias.dump_expected_time,
                ):
                    if len(temp_idle_list.points) > 0:
                        self.stats.list_of_idle_times.append(temp_idle_list)
                        temp_idle_list = Points_times()  # Re-initialize
                # Then check if there are other paramters saying we are not idle
                elif (
                    speed_kmh > self.criterias.idle_speed_limit
                ):  # Could be merged with above if stays like this.
                    if len(temp_idle_list.points) > 0:
                        self.stats.list_of_idle_times.append(temp_idle_list)
                        temp_idle_list = Points_times()  # Re-initialize
                elif (
                    speed_kmh < self.criterias.idle_speed_limit
                    and self.criterias.avg_speed_limit < current_avg_speed
                ):
                    temp_idle_list.points.append((current_pos.lat, current_pos.lon))
                    temp_idle_list.times.append(current_time)
                    if (
                        i == len(self.stats.all_positions[1:]) - 1
                    ):  # i.e. last iteration
                        self.stats.list_of_idle_times.append(temp_idle_list) """
        self.stats.day_stats_set = True
        print("Finished!")

    def idle_report(self):
        # Si noe om sum av tid idle
        # Lage plot av hvor idle
        # Hvor mange ganger idle?
        total_time_idle_seconds = sum(
            [
                (l.times[-1] - l.times[0]).total_seconds()
                for l in self.stats.list_of_idle_times
            ]
        )
        print("**************")
        print("Machine was idle ", len(self.stats.list_of_idle_times), " times.")
        print(
            "Idle for a total of (HH:MM:SS): ",
            str(timedelta(seconds=total_time_idle_seconds)),
        )
        print(
            "The average idle time per trip is: ",
            str(timedelta(seconds=(total_time_idle_seconds / len(self.machine.trips)))),
        )
        print("Heatmap of places it was idle.")

        # Create a map centered at the mean of all coordinates, with heatmap
        all_idle_points = [l.points for l in self.stats.list_of_idle_times]
        all_idle_points = [item for sublist in all_idle_points for item in sublist]
        points_center = np.mean(all_idle_points, axis=0)
        m = L.Map(center=(points_center[0], points_center[1]), zoom=10)
        # Add markers for each cluster center to the map
        heatmap = L.Heatmap(locations=all_idle_points, radius=10)
        m.add_layer(heatmap)
        # Display the map
        display(m)

    def idle_time_plot(self):
        # Plots, not wanted when a lot of data
        # Create subplots with 2 rows and 1 column
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        # Add the first line plot to the first subplot
        fig.add_trace(
            go.Scatter(
                x=self.stats.day_times,
                y=self.stats.day_speeds,
                mode="lines",
                name="Speed",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.stats.load.times,
                y=[0 for a in self.stats.load.times],
                mode="markers",
                marker=dict(symbol="cross", size=10, color="red"),
                name="Load actual",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.stats.dump.times,
                y=[0 for a in self.stats.dump.times],
                mode="markers",
                marker=dict(symbol="cross", size=10, color="green"),
                name="Dump actual",
            ),
            row=1,
            col=1,
        )

        # Add the second line plot to the second subplot
        fig.add_trace(
            go.Scatter(
                x=self.stats.day_times,
                y=np.cumsum(self.stats.day_dists),
                mode="lines",
                name="Cumulative distance",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.stats.load.times,
                y=[0 for a in self.stats.load.times],
                mode="markers",
                marker=dict(symbol="cross", size=10, color="red"),
                name="Load actual",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=self.stats.dump.times,
                y=[0 for a in self.stats.dump.times],
                mode="markers",
                marker=dict(symbol="cross", size=10, color="green"),
                name="Dump actual",
            ),
            row=2,
            col=1,
        )

        # Add the third line plot
        for idle_time in self.stats.list_of_idle_times:
            fig.add_trace(
                go.Scatter(
                    x=idle_time.times, y=[0 for a in idle_time.times], mode="lines"
                ),
                row=3,
                col=1,
            )

        # Update layout settings for both subplots
        fig.update_layout(
            title=str(
                "Subplots of Speeds and cumulative distance, machine_id: "
                + str(self.machine.machine_id)
            ),
            xaxis_title="Timestamp",
            showlegend=True,
        )

        fig.show()
        fig.write_html("./data/output_html/idle_speed_dist.html")


if __name__ == "__main__":
    day = "04-06-2022"  # MM-DD-YYYY
    # print(Automated_load_dump_for_machine.__doc__)
    automated_for_given_machine = Automated_load_dump_for_machine(day, 39)
    # automated_for_given_machine.predict_loaddump()
    # automated_for_given_machine.prediction_time_plot()
    # automated_for_given_machine.prediction_gantt_plot()
    automated_for_given_machine.find_idle_time()
    automated_for_given_machine.idle_time_plot()
    automated_for_given_machine.idle_report()

# %%
