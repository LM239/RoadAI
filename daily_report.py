#%%
import dataloader
from typing import Literal
from tqdm import tqdm
from datetime import datetime, timedelta
from pydantic import BaseModel
from schemas import Position, Trip
import geopy.distance


class Points_times(BaseModel):
    #Appendable object to store positions with datetimes
    points: list[tuple[float, float]] = []  # Latlons
    times: list[datetime] = []              # Datetimes

class Idle_machine(BaseModel):
    machine_id: str | int
    trips: list[Trip]
    load: Points_times = Points_times()  # Load points and times
    dump: Points_times = Points_times()  # Dump points and times
    list_of_idle_times: list[Points_times]
    total_idle_seconds: float

class Idle_machines(BaseModel):
    list_of_idle_machines: list[Idle_machine] = []

class Stats(BaseModel):  
    # Store relevant data on a machine
    all_positions: list[Position] = []  # Positions recorded during a day
    load: Points_times = Points_times()  # Load points and times
    dump: Points_times = Points_times()  # Dump points and times
    day_speeds: list[float] = []  # Speeds
    day_dists: list[float] = []  # Distances between each recording
    day_times: list[datetime] = []  # Timestamp for two above lists
    inner_prods: list[float] = []  # Inner product of consecutive normalized vectors
    list_of_idle_times: list[Points_times] = []  # List of all times idle during a day 
    

class MachineSummary:

    def __init__(self, trips, machine_nb: int) -> None:
        # Loading gps data for selected day and day before

        self.machine = trips._machines[machine_nb]
        self.stats = Stats()

        all_pos = [trip.positions for trip in self.machine.trips]
        self.stats.all_positions = [item for sublist in all_pos for item in sublist]
        self.stats.load.points = [trip.load_latlon for trip in self.machine.trips]
        self.stats.load.times = [
            trip.positions[0].timestamp for trip in self.machine.trips
        ]
        self.stats.dump.points = [trip.dump_latlon for trip in self.machine.trips]

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

    def find_idle_time(self):
        # A list containing idle periods before passed to object
        temp_idle_list = Points_times()

        # We start processing. Are going to iterate over all positions, from first to last
        for i in range(1, len(self.stats.all_positions[1:])):
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

                # Add the speed to a list for entire day
                self.stats.day_speeds.append(speed_kmh)

                # Add the distance (km) between the two timestamps
                self.stats.day_dists.append(meters_driven / 1000)

                # Add the timestamp for the two above values
                self.stats.day_times.append(current_pos.timestamp)

                if speed_kmh < 5: #Speed limit - can be changed as user want
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

class DailyReport:

    def __init__(self, day: str) -> None:

        # Loading gps data for selected day
        self.trips = dataloader.TripsLoader(day)

        #Initializing Idle_machine
        self.idle_machines = Idle_machines()
    
    #Function that computes idle times of choosen machine types for selected day
    def compute_idle_times(self, machine_type: Literal['Truck', 'Dumper', 'Tippbil']):

        print("Computing idle times for ", machine_type)
        for machine_id in tqdm(self.trips._machines.keys()):
            if self.trips._machines[machine_id].machine_type == machine_type:
                temp_automated = MachineSummary(self.trips, machine_id)
                temp_automated.find_idle_time()
                temp_total_time_idle_seconds = sum(
                    [
                        (item.times[-1] - item.times[0]).total_seconds()
                        for item in temp_automated.stats.list_of_idle_times
                    ]
                )

                self.idle_machines.list_of_idle_machines.append(Idle_machine(
                        machine_id = temp_automated.machine.machine_id,
                        trips = temp_automated.machine.trips,
                        load = temp_automated.stats.load,
                        dump = temp_automated.stats.dump,
                        list_of_idle_times = temp_automated.stats.list_of_idle_times,
                        total_idle_seconds = temp_total_time_idle_seconds)
                    )
        print("Finished!")


if __name__ == "__main__":
    day = "04-06-2022"  # MM-DD-YYYY
    # Here we test our function
    daily_report = DailyReport(day)

    choosen_machine_type = 'Truck'
    daily_report.compute_idle_times(choosen_machine_type)

# %%
