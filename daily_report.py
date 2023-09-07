import helper_functions.dataloader as dataloader
from typing import Literal
from tqdm import tqdm
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px
import ipyleaflet as L
import numpy as np
from helper_functions.interactive_map import InteractiveMap
from IPython.display import display, IFrame


class DailyReport:
    """
    Allows for insight into idle time, mass moved by machine and more.

    Args
    ------
    day: String specyfing day we want to look at, in format MM-DD-YYYY
    gps_data_dir: Directory path for gps data

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

    def __init__(self, day: str, gps_data_dir="data/GPSData") -> None:

        # Loading gps data for selected day
        self.trips = dataloader.TripsLoader(day, gps_data_dir)

        self.productivity = {}
        self.day = day
        self.datetime_intervals = []

        self.interactive_map = InteractiveMap(self.trips)
        for machine_id in self.trips._machines.keys():
            self.trips._machines[machine_id].all_positions
            self.trips._machines[machine_id].all_loads
            self.trips._machines[machine_id].all_dumps

    # Function that computes idle times of choosen machine types for selected day
    def compute_idle_times(self, machine_type: Literal['Truck', 'Dumper', 'Tippbil']):

        print("Computing idle times for ", machine_type)

        for machine_id, machine in tqdm(self.trips._machines.items()):
            # for machine in tqdm(self.machine_info):
            if machine.machine_type == machine_type:
                machine.list_of_idle_times
        print("Finished!")

    # Function that prepares for plotting of aggregated number of idle machines throughout day
    def aggregated_idle_timeline(self):

        machine_keys = list(self.trips._machines.keys())
        # First machines first timestamp
        first_timestamp = self.trips._machines[machine_keys[0]
                                               ].trips[0].positions[0].timestamp
        # First machines last timestamp
        last_timestamp = self.trips._machines[machine_keys[0]
                                              ].trips[-1].positions[-1].timestamp
        for machine_id, machine in self.trips._machines.items():
            if machine.trips[0].positions[0].timestamp < first_timestamp:
                first_timestamp = machine.trips[0].positions[0].timestamp
            if machine.trips[-1].positions[-1].timestamp > last_timestamp:
                last_timestamp = machine.trips[-1].positions[-1].timestamp

        # Create a list of timestamps throughout day
        current_datetime = first_timestamp

        # Do not want to look at idle machines overnight
        while current_datetime < last_timestamp and current_datetime.hour < 23:
            self.datetime_intervals.append(current_datetime)
            # This could be a parameter
            current_datetime += timedelta(minutes=2)

        self.nb_of_idle_machines = [0 for i in self.datetime_intervals]
        self.nb_of_machines_in_action = [0 for i in self.datetime_intervals]
        self.nb_of_idle_waiting_for_load = [0 for i in self.datetime_intervals]
        self.nb_of_idle_waiting_for_dump = [0 for i in self.datetime_intervals]

        # Now have a list of times, and list of machines
        for i in range(len(self.datetime_intervals)):
            time = self.datetime_intervals[i]
            for machine_id, machine in self.trips._machines.items():
                if machine.trips[0].positions[0].timestamp < time < machine.trips[-1].positions[-1].timestamp:
                    self.nb_of_machines_in_action[i] += 1
                for it in machine.list_of_idle_times:
                    if it[0].timestamp < time < it[-1].timestamp:
                        self.nb_of_idle_machines[i] += 1

                        # Check if we are waiting for load or dump
                        # Highest possible value
                        smallest_time_above = machine.trips[-1].positions[-1].timestamp
                        waiting_for_load = True

                        for lt in [point.timestamp for point in machine.all_loads]:
                            if it[0].timestamp < lt < smallest_time_above:
                                smallest_time_above = lt
                        for dt in [point.timestamp for point in machine.all_dumps]:
                            if it[0].timestamp < dt < smallest_time_above:
                                smallest_time_above = dt
                                waiting_for_load = False

                        if waiting_for_load:
                            self.nb_of_idle_waiting_for_load[i] += 1
                        else:
                            self.nb_of_idle_waiting_for_dump[i] += 1
                        break

    # Function that plots number of idle machines throughout day
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
        # fig.write_html("./data/output_html/idle_timeline.html")

    # Function that plot map of position of machines at peak times during day
    def plot_peak_times(self, nb_of_plots: int, static=False):
        # Forces you to run aggregated_idle_timeline first
        if not len(self.datetime_intervals) > 0:
            self.aggregated_idle_timeline()

        #Want to sort times
        # Combine the two lists into a list of tuples
        idle_list = self.nb_of_idle_machines.copy()
        time_list = self.datetime_intervals.copy()
        combined_lists = list(zip(idle_list, time_list))

        # Sort the combined list based on the values in list1
        sorted_combined_lists = sorted(combined_lists, key=lambda x: x[0], reverse=True)

        # Extract the sorted lists
        idle_list_sorted = [x[0] for x in sorted_combined_lists]
        time_list_sorted = [x[1] for x in sorted_combined_lists]

        #Want to plot the threshold highest values
        for i in range(nb_of_plots):
            list_of_positions = []
            list_of_load_waiting = []
            time = time_list_sorted[i]
            print("At: ", time)
            print("Idle machines: ", idle_list_sorted[i])
            for machine_id, machine in self.trips._machines.items():
                for it in machine.list_of_idle_times:
                    if it[0].timestamp < time < it[-1].timestamp:
                        # Assuming its not moving a lot during this interval
                        list_of_positions.append((it[0].lat, it[0].lon))
                        # Check if we are waiting for load or dump
                        # Highest possible value
                        smallest_time_above = machine.trips[-1].positions[-1].timestamp
                        waiting_for_load = True

                        for lt in [point.timestamp for point in machine.all_loads]:
                            if it[0].timestamp < lt < smallest_time_above:
                                smallest_time_above = lt
                        for dt in [point.timestamp for point in machine.all_dumps]:
                            if it[0].timestamp < dt < smallest_time_above:
                                smallest_time_above = dt
                                waiting_for_load = False

                        list_of_load_waiting.append(waiting_for_load)
                        break
            
            # Create a map centered at the mean of all coordinates, with heatmap
            points_center = np.mean(list_of_positions, axis=0)
            m5 = L.Map(center=(points_center[0], points_center[1]), zoom=12)
            for k in range(len(list_of_positions)):
                if list_of_load_waiting[k]:
                    load_icon = L.Icon(
                        icon_url='https://cdn-icons-png.flaticon.com/512/2716/2716797.png', icon_size=[32, 32], icon_anchor=[16, 16])
                    load_mark = L.Marker(
                        location=list_of_positions[k], icon=load_icon, rotation_angle=0, rotation_origin='22px 94px')
                    m5.add_layer(load_mark)
                else:
                    dump_icon = L.Icon(
                        icon_url='https://cdn-icons-png.flaticon.com/512/1435/1435320.png', icon_size=[32, 32], icon_anchor=[16, 16])
                    dump_mark = L.Marker(
                        location=list_of_positions[k], icon=dump_icon, rotation_angle=0, rotation_origin='22px 94px')
                    m5.add_layer(dump_mark)
            legend = L.LegendControl({},name=f"Time: {time.strftime('%m/%d/%Y, %H:%M:%S')} \n Idle machines : {idle_list_sorted[i]}")
            legend.position = "topright"  # Set position
            m5.add_control(legend)
            # Display the map
            if static:
                # STATIC VERSION OF INTERACTIVE MAP FOR HTML OUTPUT IWITH CURRENT TEXT
                #text = IHTML(str(time))
                m5.save(f'public_data/static_map/peak_idle_map{i}.html', title='PeakIdle')
                #m6 = IHTML(filename = f'public_data/static_map/peak_idle_map{time}.html')
                #display(m6, text)
                display(IFrame(src=f'public_data/static_map/peak_idle_map{i}.html', width=1000, height=600))
            else:    
                display(m5)
            # m.save('./data/output_html/my_map.html',
            #       title='PeakTime position and status')

    # Function that plots heatmap of all idle times for day
    def plot_idle_heatmap(self, static=False):

        list_of_idle_positions = []
        for machine_id, machine in self.trips._machines.items():
            temp_list = [
                item for sublist in machine.list_of_idle_times for item in sublist]
            temp_pos = [(pos.lat, pos.lon) for pos in temp_list]
            list_of_idle_positions.append(temp_pos)

        list_of_idle_positions = [
            l for sublist in list_of_idle_positions for l in sublist]
        points_center = np.mean(list_of_idle_positions, axis=0)
        m10 = L.Map(center=(points_center[0], points_center[1]), zoom=12)
        # Add markers for each cluster center to the map
        heatmap = L.Heatmap(locations=list_of_idle_positions, radius=10)
        m10.add_layer(heatmap)
        legend = L.LegendControl({},name=f"Day: {'/'.join(self.day.split('-'))}")
        legend.position = "topright"  # Set position
        m10.add_control(legend)
        # Display the map
        display(m10)
        if static:
            # STATIC VERSION OF INTERACTIVE MAP FOR HTML OUTPUT IWITH CURRENT TEXT
            #text = IHTML(str(time))
            m10.save('public_data/static_map/peak_idle_heatmap.html', title='PeakIdle')
            display(IFrame(src = 'public_data/static_map/peak_idle_heatmap.html', width=1000, height=600))
        else:    
            display(m10)
        # m.save('./data/output_html/heatmap_idle.html',
        #       title='Heatmap idle times')

    # Function that computes productivity as tons/hr (cited paper)
    def compute_productivity(self):

        # We will look at every trip of every choosen machine type
        # Except for last trip, as that is recorded in many different ways
        for mass_type in ['Stone', 'Equipment', 'Soil', '4']:

            self.productivity[mass_type] = {}

            for machine_id, machine in self.trips._machines.items():

                time_list = []
                mass_list = []
                for index, trip in enumerate(machine.trips):

                    if index < len(machine.trips)-1:  # Avoid last trip
                        if trip.load == mass_type:
                            # Want it in hours
                            time_list.append((trip.duration/60.0))
                            # We assume fully loaded
                            mass_list.append(trip.quantity)

                if sum(time_list) > 0:
                    self.productivity[mass_type][machine.machine_id] = sum(
                        mass_list)/sum(time_list)

    # Function that plots productivity
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
    choosen_machine_type = 'Truck'  # Truck | Dumper
    # Here we test our function
    # A bigger demonstration can be found in daily_report_demo notebook
    daily_report = DailyReport(day)
    daily_report.compute_idle_times(choosen_machine_type)
    daily_report.aggregated_idle_timeline()
    daily_report.plot_aggregated_idle_timeline()
