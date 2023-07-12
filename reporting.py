#%%
import matplotlib.pyplot as plt
from dataloader import TripsLoader
from schemas import Machine, Trip
from typing import Iterable, Literal
import numpy as np
import geopy.distance
import math
from scipy.interpolate import interp1d

LOAD_TYPES: list[Literal['Stone', 'Equipment', 'Soil', '4']] = ['Stone', 'Equipment', 'Soil', '4']
class DailyReport:
    def __init__(self, machines) -> None:
        self.machines = machines
    
    def complete_analysis(self):
        self.analyze(self.machines)
    
    @staticmethod
    def analyze(loader: TripsLoader):
        DailyReport.plot_load_quantities(loader.machines.values())
        DailyReport.plot_load_trips(loader.machines.values())
        DailyReport.plot_trip_counts(loader)
        DailyReport.plot_trip_length(loader)
        DailyReport.plot_trip_duration(loader)

    @staticmethod
    def analyze_machine(loader: TripsLoader, machine_id: int):
        DailyReport.plot_speed(loader, machine_id)
        DailyReport.plot_trip_separation(loader, machine_id)

    @staticmethod
    def plot_load_quantities(machines: Iterable[Machine]):
        values = []
        for load in LOAD_TYPES:
            values.append(sum(trip.quantity for machine in machines for trip in machine.trips_dict[load]))
        print(values)
        # creating the bar plot
        plt.bar(LOAD_TYPES, values, color ='maroon',
                width = 0.4)
        
        plt.xlabel("Load type")
        plt.ylabel("Quantity")
        plt.title("Title")
        plt.show()
    
    @staticmethod
    def plot_load_trips(machines: Iterable[Machine]):
        values = []
        for load in LOAD_TYPES:
            values.append(sum(len(machine.trips_dict[load]) for machine in machines))
        print(values)
        # creating the bar plot
        plt.bar(LOAD_TYPES, values, color ='maroon',
                width = 0.4)
        
        plt.xlabel("Load type")
        plt.ylabel("Trips")
        plt.title("Title")
        plt.show()

    @staticmethod
    def plot_trip_counts(loader: TripsLoader):
        machines = loader.sorted_machines('trip_count')
        # creating the bar plot
        plt.bar([str(machine.machine_id) for machine in machines], [len(machine.trips) for machine in machines], color ='maroon',
                width = 0.25)
        
        plt.xlabel("Machine_id")
        plt.ylabel("Numberof Trips")
        plt.title("Title")
        plt.show()

    @staticmethod
    def plot_trip_length(loader: TripsLoader):
        machines = loader.sorted_machines('trip_count')
        # creating the bar plot
        plt.bar([str(machine.machine_id) for machine in machines], [machine.total_length for machine in machines], color ='maroon',
                width = 0.25)
        
        plt.xlabel("Machine_id")
        plt.ylabel("Total length of trips")
        plt.title("Title")
        plt.show()

    @staticmethod
    def plot_trip_duration(loader: TripsLoader):
        machines = loader.sorted_machines('trip_length')
        print(len(machines))
        # creating the bar plot
        plt.bar([str(machine.machine_id) for machine in machines], [machine.total_duration for machine in machines], color ='maroon',
                width = 0.25)
        
        plt.xlabel("Machine_id")
        plt.ylabel("Total duration of trips (minutes)")
        plt.title("Title")
        plt.show()

    @staticmethod
    def plot_trip_separation(loader: TripsLoader, machine_id: int):
        machine = loader.machines[machine_id]

        time_diff = np.zeros(len(machine.trips) - 1)
        pos_diff = np.zeros(len(machine.trips) - 1)
        
        for i in range(len(machine.trips) - 1):
            curr_trip, next_trip = machine.trips[i:i + 2]
            last_pos, first_pos = (curr_trip.positions[-1], next_trip.positions[0])
            time_diff[i] = (first_pos.timestamp - last_pos.timestamp).seconds
            pos_diff[i] = geopy.distance.geodesic((last_pos.lat, last_pos.lon), (first_pos.lat, first_pos.lon)).km

        x = np.arange(len(machine.trips) - 1)
        fig,ax = plt.subplots()
        # make a plot
        ax.plot(x,
                time_diff,
                color="red", 
                marker="o")
        # set x-axis label
        ax.set_xlabel("Trip", fontsize = 14)
        # set y-axis label
        ax.set_ylabel("seconds_difference",
                    color="red",
                    fontsize=14)
        # twin object for two different y-axis on the sample plot
        ax2=ax.twinx()
        # make a plot with different y-axis using second axis object
        ax2.plot(x, pos_diff,color="blue",marker="o")
        ax2.set_ylabel("KM difference",color="blue",fontsize=14)
              
        plt.show()

    @staticmethod
    def plot_speed(loader: TripsLoader, machine_id: int):
        machine = loader.machines[machine_id]
        # creating the bar plot
        
        machine.trips = sorted(machine.trips, key=lambda trip: trip.start_date)

        curr_dist = 0
        cumul_dist = np.zeros(1440)
        start_time = machine.trips[0].positions[0].timestamp
        curr_i = start_time.minute + start_time.hour * 60
        curr_day = start_time.day
        print(curr_i)
        last_pos = machine.trips[0].positions[0]
        for trip in machine.trips:
            for position in trip.positions:
                new_i = position.timestamp.minute + position.timestamp.hour * 60
                if math.isnan(new_i):
                    print(trip.trip_id)
                    continue
                curr_dist += geopy.distance.geodesic((last_pos.lat, last_pos.lon), (position.lat, position.lon)).km
                last_pos = position
                if curr_i > new_i:
                    print(curr_i, new_i, position.timestamp, curr_day)
                if curr_i != new_i:
                    cumul_dist[new_i] = curr_dist
                    curr_i = new_i
                    
        idx = np.nonzero(cumul_dist)
        x = np.arange(1440)
        f = interp1d(x[idx], cumul_dist[idx], fill_value="extrapolate")
        

        cumul_dist = f(x)
        plt.plot(cumul_dist)
        
        plt.xlabel("Minute")
        plt.ylabel("Cumulative distance travelled")
        plt.title("Title")
        plt.show()

        plt.plot(np.diff(cumul_dist) * 60)
        
        plt.xlabel("Minute")
        plt.ylabel("Avg speed per minute")
        plt.title("Title")
        plt.show()
if __name__ == "__main__":
    loader = TripsLoader('03-07-2022')
    # DailyReport.analyze(loader)
    DailyReport.analyze_machine(loader, 4)

# %%
