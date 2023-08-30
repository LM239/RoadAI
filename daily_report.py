#%%
import dataloader
from typing import Literal
from tqdm import tqdm
from datetime import datetime, timedelta
from pydantic import BaseModel
from schemas import Position, Trip
import geopy.distance
import plotly.graph_objects as go
import plotly.express as px
import ipyleaflet as L
import numpy as np


class Points_times(BaseModel):
    """
    An appendable class for GPS points and times.
    """
    #Appendable object to store positions with datetimes
    points: list[tuple[float, float]] = []  # Latlons
    times: list[datetime] = []              # Datetimes

class Idle_machine(BaseModel):
    """
    Keep values for a single machine that is idle at least once during a day.
    """
    machine_id: str | int
    trips: list[Trip]
    load: Points_times = Points_times()  # Load points and times
    dump: Points_times = Points_times()  # Dump points and times
    list_of_idle_times: list[Points_times]
    total_idle_seconds: float

class Idle_machines(BaseModel):
    """
    A list of idle machines
    """
    list_of_idle_machines: list[Idle_machine] = []

class Stats(BaseModel):  
    """
    Relevant and useful information about a single machine for a selected day
    """
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
    """
    Allows for computation of idle time for a single machine and trips for that machine on a selected day.
    """

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
    """
    Allows for insight into idle time, mass moved by machine and more.

    Args
    ------
    day: String specyfing day we want to look at, in format MM-DD-YYYY

    Attributes
    ----------
    trips : All trips for given day
    idle_machines: List of Idle_machines
    productivity: Dictionary to track productivity of machines
    datetime_intervals: List with interval of times where we have active machines
    nb_of_idle_machines: nb_of_idle_machines
    nb_of_machines_in_action: List of number of machines in action
    nb_of_idle_waiting_for_load: List of number of idle machines waiting to load
    nb_of_idle_waiting_for_dump: List of number of idle machines waiting to dump

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
    def __init__(self, day: str) -> None:

        # Loading gps data for selected day
        self.trips = dataloader.TripsLoader(day)

        #Initializing Idle_machine - object that keeps track of when and where machines are idle
        self.idle_machines = Idle_machines()

        self.productivity = {}
    
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
    
    #Function that prepares for plotting of aggregated number of idle machines throughout day
    def aggregated_idle_timeline(self):
        
        
        first_timestamp = self.idle_machines.list_of_idle_machines[0].trips[0].positions[0].timestamp #First machines first timestamp
        last_timestamp = self.idle_machines.list_of_idle_machines[0].trips[-1].positions[-1].timestamp #First machines last timestamp
        for machine in self.idle_machines.list_of_idle_machines:
            if machine.trips[0].positions[0].timestamp < first_timestamp:
                first_timestamp = machine.trips[0].positions[0].timestamp
            if machine.trips[-1].positions[-1].timestamp > last_timestamp:
                last_timestamp  = machine.trips[-1].positions[-1].timestamp
        
        #Create a list of timestamps throughout day
        current_datetime = first_timestamp
        self.datetime_intervals = []

        while current_datetime < last_timestamp and current_datetime.hour < 23: #Do not want to look at idle machines overnight
            self.datetime_intervals.append(current_datetime)
            current_datetime += timedelta(minutes=2) #This could be a parameter
        
        self.nb_of_idle_machines = [0 for i in self.datetime_intervals]
        self.nb_of_machines_in_action = [0 for i in self.datetime_intervals]
        self.nb_of_idle_waiting_for_load = [0 for i in self.datetime_intervals]
        self.nb_of_idle_waiting_for_dump = [0 for i in self.datetime_intervals]
   
        #Now have a list of times, and list of machines
        for i in range(len(self.datetime_intervals)):
            time = self.datetime_intervals[i]
            for m in self.idle_machines.list_of_idle_machines:
                if m.trips[0].positions[0].timestamp < time < m.trips[-1].positions[-1].timestamp:
                   self.nb_of_machines_in_action[i] += 1 
                for it in m.list_of_idle_times:
                    if it.times[0] < time < it.times[-1]:
                        self.nb_of_idle_machines[i] += 1
                        
                        #Check if we are waiting for load or dump
                        smallest_time_above = m.trips[-1].positions[-1].timestamp #Highest possible value
                        waiting_for_load = True
                        
                        for lt in m.load.times:
                            if  it.times[0] < lt < smallest_time_above:
                                smallest_time_above = lt
                        for dt in m.dump.times:
                            if it.times[0] < dt < smallest_time_above:
                                smallest_time_above = dt
                                waiting_for_load = False
                        
                        if waiting_for_load:
                            self.nb_of_idle_waiting_for_load[i] += 1
                        else:
                            self.nb_of_idle_waiting_for_dump[i] += 1
                        break
    
    #Function that plots number of idle machines throughout day
    def plot_aggregated_idle_timeline(self):

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.datetime_intervals,
            y=self.nb_of_idle_machines,
            mode='markers+lines',
            name='Machines idle'
        ))

        fig.add_trace(go.Scatter(
            x=self.datetime_intervals,
            y=self.nb_of_machines_in_action,
            mode='markers+lines',
            name='Machines in action'
        ))

        fig.add_trace(go.Scatter(
            x=self.datetime_intervals,
            y=self.nb_of_idle_waiting_for_load,
            mode='markers+lines',
            name='Waiting for load'
        ))

        fig.add_trace(go.Scatter(
            x=self.datetime_intervals,
            y=self.nb_of_idle_waiting_for_dump,
            mode='markers+lines',
            name='Waiting for dump'
        ))


        fig.update_layout(
            title='Number of concurrently idle machines',
            xaxis_title='Time',
            yaxis_title='Machines idle',
            xaxis=dict(type='date'),
            yaxis=dict(type='linear'),
        )

        fig.show()
        #fig.write_html("./data/output_html/idle_timeline.html")
    
    #Function that plot map of position of machines at peak times during day
    def plot_peak_times(self, threshold: int):

        #Froces you to run aggregated_idle_timeline first
        if not len(self.datetime_intervals) > 0 :
            self.aggregated_idle_timeline()

        
        last_val = 0 #Want to avoid plotting similar maps for same idle period
        
        for i in range(len(self.datetime_intervals)):
            list_of_positions = []
            list_of_load_waiting = []
            if self.nb_of_idle_machines[i] >= threshold and self.nb_of_idle_machines[i] > last_val:
                last_val = self.nb_of_idle_machines[i]
                #We are at or above threshold. Want to plot position of idle machines
                time = self.datetime_intervals[i]
                print("At: ", time)
                print("Idle machines: ", self.nb_of_idle_machines[i])
                for m in self.idle_machines.list_of_idle_machines:
                    for it in m.list_of_idle_times:
                        if it.times[0] < time < it.times[-1]:
                            list_of_positions.append(it.points[0])#Assuming its not moving a lot during this interval
                            #Check if we are waiting for load or dump
                            smallest_time_above = m.trips[-1].positions[-1].timestamp #Highest possible value
                            waiting_for_load = True
                            
                            for lt in m.load.times:
                                if  it.times[0] < lt < smallest_time_above:
                                    smallest_time_above = lt
                            for dt in m.dump.times:
                                if it.times[0] < dt < smallest_time_above:
                                    smallest_time_above = dt
                                    waiting_for_load = False
                            
                            list_of_load_waiting.append(waiting_for_load)
                            break
        
                # Create a map centered at the mean of all coordinates, with heatmap
                points_center = np.mean(list_of_positions, axis=0)
                m = L.Map(center=(points_center[0], points_center[1]), zoom=12)
                for k in range(len(list_of_positions)):
                    if list_of_load_waiting[k]:
                        load_icon = L.Icon(icon_url='https://cdn-icons-png.flaticon.com/512/2716/2716797.png', icon_size=[32, 32], icon_anchor=[16,16])
                        load_mark = L.Marker(location=list_of_positions[k], icon=load_icon, rotation_angle=0, rotation_origin='22px 94px')
                        m.add_layer(load_mark)
                    else:
                        dump_icon = L.Icon(icon_url='https://cdn-icons-png.flaticon.com/512/1435/1435320.png', icon_size=[32, 32], icon_anchor=[16,16])
                        dump_mark = L.Marker(location=list_of_positions[k], icon=dump_icon, rotation_angle=0, rotation_origin='22px 94px')
                        m.add_layer(dump_mark)
                # Display the map
                display(m)
                m.save('./data/output_html/my_map.html', title='PeakTime position and status')
            else:
                last_val = 0

    #Function that plots heatmap of all idle times for day
    def plot_idle_heatmap(self):

        list_of_idle_positions = []
        for idle_machine in self.idle_machines.list_of_idle_machines:
            temp_pos = [t.points for t in idle_machine.list_of_idle_times]
            list_of_idle_positions.append(temp_pos)

        
        list_of_idle_positions = [l for sublist in list_of_idle_positions for l in sublist]
        list_of_idle_positions = [l for sublist in list_of_idle_positions for l in sublist]
        points_center = np.mean(list_of_idle_positions, axis=0)
        m = L.Map(center=(points_center[0], points_center[1]), zoom=12)
        # Add markers for each cluster center to the map
        heatmap = L.Heatmap(locations=list_of_idle_positions, radius=10)
        m.add_layer(heatmap)
        # Display the map
        display(m)
    
    #Function that computes productivity as tons/hr (cited paper)
    def compute_productivity(self):
        
        #We will look at every trip of every choosen machine type
        #Except for last trip, as that is recorded in many different ways
        for mass_type in ['Stone', 'Equipment', 'Soil', '4']:
            
            self.productivity[mass_type] = {}

            for machine in self.trips._machines.keys():

                temp_machine = self.trips._machines[machine]
                
                time_list = []
                mass_list = []
                for index, trip in enumerate(temp_machine.trips):
                    
                    if index < len(temp_machine.trips)-1: #Avoid last trip
                        if trip.load == mass_type:
                            time_list.append((trip.positions[-1].timestamp-trip.positions[0].timestamp).total_seconds())
                            mass_list.append(trip.quantity) #We assume fully loaded

                if sum(time_list) > 0:
                    time_list = [seconds/(60*60) for seconds in time_list] #Want it in hours
                    self.productivity[mass_type][temp_machine.machine_id] = sum(mass_list)/sum(time_list)
                    
    #Function that plots productivity
    def plot_productivity(self):

        for mass_type in ['Stone', 'Equipment', 'Soil', '4']:

            temp_dict = self.productivity[mass_type]
            if bool(temp_dict):
                # Extract keys and values from the dictionary
                keys = list(temp_dict.keys())
                values = list(temp_dict.values())

                # Create a bar plot using Plotly Express
                fig = px.bar(x=keys, y=values)

                # Customize the layout with title and axis titles
                fig.update_layout(
                    title=f"Bar Plot of productivity for mass type: {mass_type}",
                    xaxis_title="Machine id",
                    yaxis_title="tons/hr"
                )
                fig.update_xaxes(type='category')
                fig.update_xaxes(categoryorder='total descending')

                # Show the bar plot
                fig.show()




if __name__ == "__main__":
    day = "04-06-2022"  # MM-DD-YYYY
    choosen_machine_type = 'Truck' # Truck | Dumper
    # Here we test our function
    daily_report = DailyReport(day)

    #daily_report.compute_idle_times(choosen_machine_type)
    #daily_report.aggregated_idle_timeline()
    #daily_report.plot_aggregated_idle_timeline()
    #daily_report.plot_peak_times(12)
    #daily_report.plot_idle_heatmap()
    daily_report.compute_productivity()
    daily_report.plot_productivity()

# %%
