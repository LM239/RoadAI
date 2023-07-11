#%%
import matplotlib.pyplot as plt
from dataloader import TripsLoader
from schemas import Machine, Trip
from typing import Iterable, Literal

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
        plt.ylabel("Quantity")
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
if __name__ == "__main__":
    loader = TripsLoader('03-07-2022')
    DailyReport.analyze(loader)

# %%
