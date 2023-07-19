# %%
from dataloader import TripsLoader
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import geopy.distance
import ipyleaflet as L
from ipywidgets import Layout
from datetime import timedelta
from sklearn.cluster import KMeans
from tqdm import tqdm
from schemas import Trip


class AutomatedDumpPoints:
    def __init__(self, datestring: str, cluster_datestring: str) -> None:

        print("Loading trips for given day.")
        self.day_loader = TripsLoader(datestring)

        print("Generating clusters.")
        self.cluster_loader = TripsLoader(cluster_datestring)
        self.generate_clusters()

        print("Prediction of dump points beginning:")
        self.predict_dumps_day()

        print("Analysis of prediction:")
        self.prediction_analysis()

    def generate_clusters(self):
        """
        Generate clusters for all dump points on day.
        Want to build out with different clustering algorithms.
        """

        all_dump_positions_for_day_before = []

        for machine_number in self.cluster_loader._machines.keys():
            temp_machine = self.cluster_loader._machines[machine_number]
            all_dump_positions_for_day_before.append(
                [trip.dump_latlon for trip in temp_machine.trips])

        all_dump_positions_for_day_before = [
            item for sublist in all_dump_positions_for_day_before for item in sublist]

        self.prev_day_dump_points = np.array(all_dump_positions_for_day_before)

        # Hardcoded at the moment
        optimal_K = 30

        # Fit the KMeans model with the optimal K value
        kmeans = KMeans(n_clusters=optimal_K, random_state=42, n_init='auto')
        kmeans.fit(self.prev_day_dump_points)

        # Get the coordinates of the cluster centers for the optimal K value
        self.cluster_centers = kmeans.cluster_centers_

    def predict_dump_trip(self, trip: Trip):
        """
        Predict dumps for a given trip
        """

        trip_speeds = []
        trip_dists = []
        time_keeper = []
        # Just give it some value
        time_of_dump = trip.positions[0].timestamp
        # Initial value, should maybe be something else
        predicted_dump_time = trip.positions[0].timestamp
        predicted_dump_latlon = (
            trip.positions[0].lat, trip.positions[0].lon)  # Initialize value
        entering_working_area = []
        exiting_working_area = []
        initial_lat_lon = (
            trip.positions[0].lat, trip.positions[0].lon)
        in_working_area = False
        predicted_dump = False

        for coord in self.cluster_centers:
            if geopy.distance.geodesic(coord, initial_lat_lon).m < 100:
                in_working_area = True

        for i in range(len(trip.positions[1:])):
            current_time = trip.positions[i].timestamp
            prev_time = trip.positions[i-1].timestamp

            current_lat_lon = (
                trip.positions[i].lat, trip.positions[i].lon)

            if current_lat_lon == trip.dump_latlon:
                time_of_dump = current_time

            prev_lat_lon = (
                trip.positions[i-1].lat, trip.positions[i-1].lon)

            seconds_gone = (current_time.to_pydatetime(
            )-prev_time.to_pydatetime()).total_seconds()
            meters_driven = geopy.distance.geodesic(
                prev_lat_lon, current_lat_lon).m

            if seconds_gone > 0:
                speed_kmh = (meters_driven/seconds_gone)*3.6
                trip_speeds.append(speed_kmh)
                trip_dists.append(meters_driven/1000)
                time_keeper.append(current_time)

                currently_in_working_area = False
                for coord in self.cluster_centers:
                    if geopy.distance.geodesic(coord, current_lat_lon).m < 100:
                        currently_in_working_area = True

                if not in_working_area and currently_in_working_area:
                    entering_working_area.append(current_time)
                    in_working_area = True
                elif in_working_area and not currently_in_working_area:
                    exiting_working_area.append(current_time)
                    in_working_area = False

                # Demands to predict dump point
                # want to add cumsum last 2 ish minutes to be lower than 500m for example
                if in_working_area and speed_kmh < 5 and sum(trip_dists) > 0.2 and not predicted_dump:
                    last_min_start = current_time-timedelta(minutes=1)
                    index_start_minute = None
                    for i, ts in enumerate(time_keeper):
                        if ts >= last_min_start:
                            index_start_minute = i
                            break
                    sum_over_last_minute = np.sum(
                        trip_speeds[index_start_minute:])
                    if sum_over_last_minute < 200:
                        predicted_dump_latlon = current_lat_lon
                        predicted_dump_time = current_time
                        predicted_dump = True

        plot_data = {'time_keeper': time_keeper,
                     'trip_speeds': trip_speeds,
                     'trip_dists': trip_dists,
                     'entering_working_area': entering_working_area,
                     'exiting_working_area': exiting_working_area}

        return predicted_dump_latlon, predicted_dump_time, time_of_dump, plot_data

    def predict_dumps_day(self):
        """
        Predict dumps, and save errors to a DataFrame
        """

        pred_dict = {}
        for machine_id in tqdm(self.day_loader._machines.keys()):
            some_machine = self.day_loader._machines[machine_id]
            for trip in some_machine.trips:

                predicted_dump_latlon, predicted_dump_time, time_of_dump, _ = self.predict_dump_trip(
                    trip)

                pred_dict[trip.trip_id] = {
                    "machine_type": some_machine.machine_type,
                    "machine_id": some_machine.machine_id,
                    "load": trip.load,
                    "quantity": trip.quantity,
                    "predicted_dump_latlon": predicted_dump_latlon,
                    "actual_dump_latlon": trip.dump_latlon,
                    "predicted_dump_time": predicted_dump_time,
                    "actual_dump_time": time_of_dump,
                    "second_error": (time_of_dump-predicted_dump_time).total_seconds(),
                    "meter_error": geopy.distance.geodesic(predicted_dump_latlon, trip.dump_latlon).m}

        self.predictions = pd.DataFrame(pred_dict).T

    def plot_clusters(self):
        """
        Plot the clusters and heatmaps of dumping points
        """

        # Create a map centered at the mean of all coordinates, with heatmap
        map_center = np.mean(self.prev_day_dump_points, axis=0)
        m = L.Map(center=(map_center[0], map_center[1]), zoom=10)

        # Add markers for each cluster center to the map
        for center in self.cluster_centers:
            marker = L.Marker(location=(center[0], center[1]))
            m.add_layer(marker)
        heatmap = L.Heatmap(
            locations=self.prev_day_dump_points.tolist(), radius=20)
        m.add_layer(heatmap)

        # Display the map
        display(m)

    def plot_machine_prediction(self, machine_id: int):
        """
        Plot the dump predictions for a given machine
        """

        some_machine = self.day_loader._machines[machine_id]
        for trip in some_machine.trips:

            predicted_dump_latlon, predicted_dump_time, time_of_dump, plot_data = self.predict_dump_trip(
                trip)

            # Create subplots with 2 rows and 1 column
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

            # Add the first line plot to the first subplot
            fig.add_trace(go.Scatter(x=plot_data['time_keeper'], y=plot_data['trip_speeds'],
                          mode='lines', name='Speed'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[time_of_dump], y=[0], mode='markers', marker=dict(
                symbol='cross', size=10, color='red'), name='Dump actual'), row=1, col=1)
            fig.add_trace(go.Scatter(x=[predicted_dump_time], y=[0], mode='markers', marker=dict(
                symbol='star', size=10, color='red'), name='Dump predicted'), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_data['entering_working_area'], y=[0 for e in plot_data['entering_working_area']], mode='markers', marker=dict(
                symbol='cross', size=10, color='green'), name='Entering working area'), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_data['exiting_working_area'], y=[0 for e in plot_data['exiting_working_area']], mode='markers', marker=dict(
                symbol='cross', size=10, color='yellow'), name='Exiting working area'), row=1, col=1)

            # Add the second line plot to the second subplot
            fig.add_trace(go.Scatter(x=plot_data['time_keeper'], y=np.cumsum(
                plot_data['trip_dists']), mode='lines', name='Cumulative distance'), row=2, col=1)
            fig.add_trace(go.Scatter(x=[time_of_dump], y=[0], mode='markers', marker=dict(
                symbol='cross', size=10, color='red'), name='Dump actual'), row=2, col=1)
            fig.add_trace(go.Scatter(x=[predicted_dump_time], y=[0], mode='markers', marker=dict(
                symbol='star', size=10, color='red'), name='Dump predicted'), row=2, col=1)
            fig.add_trace(go.Scatter(x=plot_data['entering_working_area'], y=[0 for e in plot_data['entering_working_area']], mode='markers', marker=dict(
                symbol='cross', size=10, color='green'), name='Entering working area'), row=2, col=1)
            fig.add_trace(go.Scatter(x=plot_data['exiting_working_area'], y=[0 for e in plot_data['exiting_working_area']], mode='markers', marker=dict(
                symbol='cross', size=10, color='yellow'), name='Exiting working area'), row=2, col=1)

            # Update layout settings for both subplots
            fig.update_layout(title=str('Subplots of Speeds and cumulative distance, trip ID: ' + trip.trip_id),
                              xaxis_title='Timestamp',
                              showlegend=True)

            # Show the plot
            fig.show()

            m = L.Map(layout=Layout(width='60%', height='700px'),
                      center=[59.95, 10.3])
            line = L.Polyline(locations=[(pos.lat, pos.lon)
                              for pos in trip.positions], color="blue", fill=False)
            start_point = L.CircleMarker(
                location=(trip.positions[0].lat, trip.positions[0].lon), color="green")
            end_point = L.CircleMarker(
                location=(trip.positions[-1].lat, trip.positions[-1].lon), color="orange")
            actual_dump_point = L.CircleMarker(
                location=trip.dump_latlon, color="red")
            predicted_dump_point = L.Marker(
                location=predicted_dump_latlon, draggable=False)
            m.add_layer(line)
            m.add_layer(start_point)
            m.add_layer(end_point)
            m.add_layer(actual_dump_point)
            m.add_layer(predicted_dump_point)
            display(m)

    def prediction_analysis(self):
        """
        Some plots to analyze predictions
        """
        fig = px.histogram(self.predictions, x="meter_error",
                           color="machine_type", nbins=500, range_x=[0, 1000])
        fig.show()

        fig = px.histogram(self.predictions, x="second_error",
                           color="machine_type", nbins=10000, range_x=[-5000, 5000])
        fig.show()


if __name__ == "__main__":
    autDump = AutomatedDumpPoints(datestring='03-08-2022',
                                  cluster_datestring='03-07-2022')

# %%
