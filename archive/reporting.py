# %%
import matplotlib.pyplot as plt
from helper_functions.dataloader import TripsLoader
from helper_functions.schemas import Machine, Trip
from typing import Iterable, Literal
import numpy as np
import geopy.distance
import math
from scipy.interpolate import interp1d

LOAD_TYPES: list[Literal['Stone', 'Equipment', 'Soil', '4']] = [
    'Stone', 'Equipment', 'Soil', '4']


class DailyReport:
    def __init__(self, machines) -> None:
        self.machines = machines

    def complete_analysis(self):
        self.analyze(self.machines)

    @staticmethod
    def analyze(loader: TripsLoader):
        """Run all analyses that apply to all vehicles for a day
        """
        DailyReport.plot_load_quantities(loader.machines.values())
        DailyReport.plot_load_trips(loader.machines.values())
        DailyReport.plot_trip_counts(loader)
        DailyReport.plot_trip_length(loader)
        DailyReport.plot_trip_duration(loader)

    @staticmethod
    def analyze_machine(loader: TripsLoader, machine_id: int):
        """Run analyses for the vehicle with the given machine_id
        """
        DailyReport.plot_speed(loader, machine_id)
        DailyReport.plot_trip_separation(loader, machine_id)

    @staticmethod
    def plot_load_quantities(machines: Iterable[Machine]):
        """Plot the combined quantity for each load type from the given Machines
        """
        values = []
        for load in LOAD_TYPES:
            values.append(
                sum(trip.quantity for machine in machines for trip in machine.trips_dict[load]))

        # creating the bar plot
        plt.bar(LOAD_TYPES, values, color='maroon',
                width=0.4)

        plt.xlabel("Load type")
        plt.ylabel("Quantity")
        plt.title("Title")
        plt.show()

    @staticmethod
    def plot_load_trips(machines: Iterable[Machine]):
        """Plot the number of trips for each load type from the given Machines"""
        values = []
        for load in LOAD_TYPES:
            values.append(
                sum(len(machine.trips_dict[load]) for machine in machines))
        print(values)
        # creating the bar plot
        plt.bar(LOAD_TYPES, values, color='maroon',
                width=0.4)

        plt.xlabel("Load type")
        plt.ylabel("Trips")
        plt.title("Title")
        plt.show()

    @staticmethod
    def plot_trip_counts(loader: TripsLoader):
        """ Plot the number of trips for each mahine id in loader
        """
        machines = loader.sorted_machines('trip_count')
        # creating the bar plot
        plt.bar([str(machine.machine_id) for machine in machines], [len(machine.trips) for machine in machines], color='maroon',
                width=0.25)

        plt.xlabel("Machine_id")
        plt.ylabel("Numberof Trips")
        plt.title("Title")
        plt.show()

    @staticmethod
    def plot_trip_length(loader: TripsLoader):
        """Plot the total length (in kilometeers) of the trips for each machine in loader
        """
        machines = loader.sorted_machines('trip_count')
        # creating the bar plot
        plt.bar([str(machine.machine_id) for machine in machines], [machine.total_length for machine in machines], color='maroon',
                width=0.25)

        plt.xlabel("Machine_id")
        plt.ylabel("Total length of trips")
        plt.title("Title")
        plt.show()

    @staticmethod
    def plot_trip_duration(loader: TripsLoader):
        """Plot the total duration (in minutes) of the trips for each machine in loader
        """
        machines = loader.sorted_machines('trip_length')
        print(len(machines))
        # creating the bar plot
        plt.bar([str(machine.machine_id) for machine in machines], [machine.total_duration for machine in machines], color='maroon',
                width=0.25)

        plt.xlabel("Machine_id")
        plt.ylabel("Total duration of trips (minutes)")
        plt.title("Title")
        plt.show()

    @staticmethod
    def plot_trip_separation(loader: TripsLoader, machine_id: int):
        """For the given machine, plot the difference in position (km) and time (s) for each pair of consecutive trips in loader
        """
        machine = loader.machines[machine_id]

        time_diff = np.zeros(len(machine.trips) - 1)
        pos_diff = np.zeros(len(machine.trips) - 1)

        for i in range(len(machine.trips) - 1):
            curr_trip, next_trip = machine.trips[i:i + 2]

            # get last post of current trip, and first pos of net trip
            last_pos, first_pos = (
                curr_trip.positions[-1], next_trip.positions[0])

            # get difference in timestamp, measured in seconds
            time_diff[i] = (last_pos.timestamp - first_pos.timestamp).seconds
            pos_diff[i] = geopy.distance.geodesic(
                (last_pos.lat, last_pos.lon), (first_pos.lat, first_pos.lon)).km

        _, ax = plt.subplots()
        # make plot 1
        ax.plot(time_diff, color="red", marker="o")
        ax.set_xlabel("Trip", fontsize=14)
        ax.set_ylabel("Seconds difference", color="red", fontsize=14)

        # twin object for two different y-axis on the sample plot
        ax2 = ax.twinx()
        ax2.plot(pos_diff, color="blue", marker="o")
        ax2.set_ylabel("KM difference", color="blue", fontsize=14)

        plt.show()

    @staticmethod
    def plot_speed(loader: TripsLoader, machine_id: int):
        """PLot the avg speed and cumulative distance travelled for the given machine
            Assumes that trips are non-overlapping 
        """
        machine = loader.machines[machine_id]
        # creating the bar plot

        machine.trips = sorted(machine.trips, key=lambda trip: trip.start_date)

        curr_dist = 0  # The current cumulative distance travelled
        # List to store the cumulative distance travelled for each minute of the day
        cumul_dist = np.zeros(1440)
        start_time = machine.trips[0].positions[0].timestamp
        curr_i = start_time.minute + start_time.hour * 60
        curr_day = start_time.day
        last_pos = machine.trips[0].positions[0]
        for trip in machine.trips:
            for position in sorted(trip.positions, key=lambda pos: pos.timestamp):
                new_i = position.timestamp.minute + position.timestamp.hour * 60
                # If a trip goes into the next day, we must extend cumul_dist
                new_day = position.timestamp.day

                # Find the distance
                curr_dist += geopy.distance.geodesic(
                    (last_pos.lat, last_pos.lon), (position.lat, position.lon)).km
                last_pos = position
                if curr_i > new_i:
                    # Some thing has gone wrong :( !! The trips are probably overlapping
                    # When new_i goes back in time, everything breaks because the cumulative distance counts future movement
                    print(curr_i, new_i, position.timestamp, curr_day)
                if new_day != curr_day:
                    # The time stamp is on the next day, so we must reset some things, and count for two days instead of one!
                    # add one day owrth of seconds for each day of differrence
                    new_i += (position.timestamp - start_time).days * 1440
                    print(
                        f"Found {(position.timestamp - start_time).days} extra day(s)")
                    # Extend our list of cumulative distance travelled
                    cumul_dist = np.concatenate((cumul_dist, np.arange(
                        (position.timestamp - start_time).days * 1440)))
                    # Reset start_time such that it is on the current day
                    start_time = position.timestamp
                    curr_day = new_day  # Set new value for the day of the month
                if curr_i != new_i:
                    # We have reached a new minute, and therefore store the cumulative distance
                    cumul_dist[new_i] = curr_dist
                    curr_i = new_i

        # Linearly interpolate the cumulative distance for minutes that were not logged by the GPS
        idx = np.nonzero(cumul_dist)
        x = np.arange(1440)
        f = interp1d(x[idx], cumul_dist[idx], fill_value="extrapolate")

        cumul_dist = f(x)  # Interpolated, cumulative distance
        plt.plot(cumul_dist)  # Plot cumulative distance

        plt.xlabel("Minute")
        plt.ylabel("Cumulative distance travelled")
        plt.title("Title")
        plt.show()

        # Plot speed in km/h
        plt.plot(np.diff(cumul_dist) * 60)

        plt.xlabel("Minute")
        plt.ylabel("Avg speed per minute")
        plt.title("Title")
        plt.show()


if __name__ == "__main__":
    loader = TripsLoader('05-07-2022')
    # DailyReport.analyze(loader)
    DailyReport.analyze_machine(loader, 1)

# %%
