from ipyleaflet import Map, Polygon, DivIcon, LayerGroup, Marker, ImageOverlay, LegendControl
import numpy as np
from helper_functions.dataloader import TripsLoader
from ipywidgets import Layout, HTML, SelectionSlider, IntSlider, Output, VBox, HBox
from scipy.spatial import ConvexHull
from sklearn.cluster import AgglomerativeClustering
from IPython.core.display import HTML as IHTML
from IPython.display import display, clear_output, IFrame
import pickle as pkl
from datetime import datetime
import os

# The `InteractiveMap` class is a Python class that represents an interactive map with overlays and
# markers for dump and load regions, and provides functionality to update the map based on user input.


class InteractiveMap:
    def __init__(self, trips: TripsLoader) -> None:
        """
        The function initializes various attributes and widgets for a class, including sliders and
        textboxes for user interaction.

        :param trips: The `trips` parameter is an instance of the `TripsLoader` class. It is used to
        load and access trip data
        :type trips: TripsLoader
        """
        self.trips: TripsLoader = trips
        self.overlays: list = []
        self.k_load = None
        self.k_dump = None
        self.machine_type: str = 'Truck'
        self.day: str = trips.day
        self.file_to_bounds_path = 'public_data/png_folder/file_to_bounds.pkl'

        self.quantity_by_region: dict = {}
        self.quantity_over_time_per_machine: dict = {}
        self.load_latlons = []
        self.dump_latlons = []
        self.load_labels = []
        self.dump_labels = []
        self.quantity = []
        self.load_type = []

        self.m: Map = Map(layout=Layout(width='60%',
                                        height='500px'),
                          center=(59.95, 10.35))

        self.k_dump_slider: IntSlider = \
            IntSlider(min=2,
                      max=10,
                      value=4,
                      description='Clusters Dump',
                      disabled=False,
                      continuous_update=False,
                      orientation='horizontal',
                      readout=True,
                      step=1,
                      layout=Layout(width='400px'),
                      style={'description_width': 'initial'})
        self.k_load_slider: IntSlider = \
            IntSlider(min=2,
                      max=10,
                      value=4,
                      description='Clusters Load',
                      disabled=False,
                      continuous_update=False,
                      orientation='horizontal',
                      readout=True,
                      step=1,
                      layout=Layout(width='400px'),
                      style={'description_width': 'initial'})

        self.machine_type_slider: SelectionSlider = \
            SelectionSlider(options=['Truck', 'Dumper'],
                            value='Truck',
                            description='Machine Type',
                            disabled=False,
                            continuous_update=False,
                            orientation='horizontal',
                            readout=True,
                            layout=Layout(
                                width='400px'),
                            style={'description_width': 'initial'})
        self.info_textbox = HTML()
        self.out = Output()

        self.k_dump_slider.observe(self.update_k_dump, names='value')
        self.k_load_slider.observe(self.update_k_load, names='value')
        self.machine_type_slider.observe(
            self.update_machine_type, names='value')

    def plot_interactive_map(self, jupyter=False):
        """
        The function plots an interactive map with sliders and a textbox for user input.
        """

        self.update_map()

        self.initial_map_overlay(jupyter=jupyter)
        # set optimal k's before any slider modifications
        self.k_dump_slider.value = self.k_dump
        self.k_load_slider.value = self.k_load

        self.update_textbox()

        vbox_layout = VBox([self.k_dump_slider, self.k_load_slider,
                           self.machine_type_slider, self.info_textbox])
        hbox_layout = HBox([self.m, vbox_layout])
        display(hbox_layout)
    
    def plot_static_map(self):
        # STATIC VERSION OF INTERACTIVE MAP FOR HTML OUTPUT IWITH CURRENT TEXT
        #clear_output()
        text = IHTML(self.info_textbox.value)
        self.m.save('public_data/static_map/interact_static_ver.html', title='MYMAP')
        map = IHTML('public_data/static_map/interact_static_ver.html')
        #mapstr = IHTML('public_data/static_map/interact_static_ver.html')
        display(map, text)
        #display(IFrame(src='public_data/static_map/interact_static_ver.html', width=1000, height=600))

    def initial_map_overlay(self, jupyter=False):
        """
        The function `initial_map_overlay` adds image overlays to a map based on specific file names and
        their corresponding bounds.
        """
        file_to_bounds = {}
        with open(self.file_to_bounds_path, 'rb') as handle:
            file_to_bounds = pkl.load(handle)

        input_date = datetime.strptime(self.day, '%m-%d-%Y')

        filenames_skaret = [e for e in file_to_bounds.keys() if e.split('_')[
            2] == 'Skaret-Orthomosaic.png']
        filenames_nordlandsdalen = [e for e in file_to_bounds.keys() if e.split('_')[
            2] == 'Nordlandsdalen-Orthomosaic.png']
        skaret_dates = [datetime.strptime(
            e.split('_')[1], '%y%m%d') for e in filenames_skaret]
        nordlandsdalen_dates = [datetime.strptime(
            e.split('_')[1], '%y%m%d') for e in filenames_nordlandsdalen]
        skaret_closest_date = self.find_closest_date(input_date, skaret_dates)
        nordlandsdalen_closest_date = self.find_closest_date(
            input_date, nordlandsdalen_dates)
        date_string_skaret = 'P07_' + \
            skaret_closest_date.strftime('%y%m%d') + '_Skaret-Orthomosaic.png'
        date_string_nordlandsdalen = 'P08_' + \
            nordlandsdalen_closest_date.strftime(
                '%y%m%d') + '_Nordlandsdalen-Orthomosaic.png'
        path_relative = 'public_data/png_folder/'
        path_host = 'https://raw.githubusercontent.com/oyste/image_host/main/docs/assets/'

        path = path_host if jupyter else path_relative

        # add skaret og nordlandsdalen overlay
        im_overlay_skaret = ImageOverlay(
            url=path + date_string_skaret, bounds=file_to_bounds[date_string_skaret])
        im_overlay_nordlandsdalen = ImageOverlay(
            url=path + date_string_nordlandsdalen, bounds=file_to_bounds[date_string_nordlandsdalen])
        self.m.add_layer(im_overlay_skaret)
        self.m.add_layer(im_overlay_nordlandsdalen)

    def find_closest_date(self, input_date, date_list):
        """
        The function finds the closest date to a given input date from a list of dates.

        :param input_date: The input_date parameter is the date for which you want to find the closest
        date in the date_list
        :param date_list: A list of dates that you want to compare with the input_date
        :return: the closest date from the given date list.
        """
        closest_date = None
        min_time_difference = None

        for date in date_list:
            time_difference = abs(date - input_date)

            if min_time_difference is None or time_difference < min_time_difference:
                min_time_difference = time_difference
                closest_date = date

        return closest_date

    def update_k_dump(self, change):
        """
        The function updates the value of the variable "k_dump" and then calls two other functions to
        update the map and textbox.

        :param change: The `change` parameter is a dictionary that contains information about the change
        that occurred. It typically has a key called `'new'` which represents the new value of the
        variable being updated
        """
        with self.out:
            self.k_dump = change['new']
            self.update_map()
            self.update_textbox()

    def update_k_load(self, change):
        """
        The function updates the value of `k_load` attribute, calls `update_map()` and
        `update_textbox()` methods.

        :param change: The `change` parameter is a dictionary that contains information about the change
        that occurred. It typically has a key called 'new' which represents the new value of the
        variable being changed
        """
        with self.out:
            self.k_load = change['new']
            self.update_map()
            self.update_textbox()

    def update_machine_type(self, change):
        """
        The function updates the machine type and adjusts the optimal amount of clusters accordingly.

        :param change: The `change` parameter is a dictionary that contains the new value for the
        machine type
        """
        with self.out:
            self.machine_type = change['new']

            # update optimal amount of clusters for new machine type
            self.k_dump = None
            self.k_load = None

            self.update_map()
            self.k_dump_slider.value = self.k_dump
            self.k_load_slider.value = self.k_load
            self.update_textbox()

    def update_map(self):
        """
        The function updates a map by removing previous overlays, plotting new polygons and markers, and
        populating the map with dumps and loads.
        """
        # remove previous overlays
        for overlay in self.overlays:
            self.m.remove_layer(overlay)
        self.overlays.clear()

        # plot new polygons and markers
        self.dump_and_load_clustering()

        self.find_quantity_moved_per_region()
        #  populate map with dumps and loads
        self.populate_map('dump')
        self.populate_map('load')

        # legend overlay
        leg = LegendControl({'Dump': '#00F', 'Load': '#F00'},
                            name='Zones', position='topright')
        self.m.add_control(leg)
        self.overlays.append(leg)

    def update_textbox(self):
        """
        The function `update_textbox` generates a summary of various statistics related to mass movement
        for a given day, including total mass moved, mass moved by load type, top mass transfer zones,
        and top workers.
        """
        title = f'<b>Day Overview, {self.day}</b><br>'

        # find total mass moved for the day
        t_mass = sum([e for e in self.quantity])
        t_mass_moved = f'Total mass moved for the day by {self.machine_type}: {round(t_mass, 1)} t<br>'
        t_mass_stone = sum(
            [q for lt, q in zip(self.load_type, self.quantity) if lt == 'Stone'])
        t_mass_soil = sum(
            [q for lt, q in zip(self.load_type, self.quantity) if lt == 'Soil'])
        t_mass_eqip = sum(
            [q for lt, q in zip(self.load_type, self.quantity) if lt == 'Equipment'])
        t_mass_other = sum(
            [q for lt, q in zip(self.load_type, self.quantity) if lt == '4'])
        load_type_info = f'Stone: {t_mass_stone} t, Soil: {t_mass_soil} t, Equipment: {t_mass_eqip} t, Other: {t_mass_other} t<br>'

        # list load to dump zones by importance (weight decending) [up to 5 instances]
        sorted_load_to_dump_zones = dict(sorted(
            self.quantity_by_region['load_to_dump'].items(), key=lambda item: item[1], reverse=True))
        important_zones_text = ''
        cnt = 0
        for k, v in sorted_load_to_dump_zones.items():
            if cnt >= 5:
                break
            important_zones_text += f'{k}: {round(v,1)} t<br>'
            cnt += 1
        important_zones_text = f'<b>Top {cnt} mass transfer zones for the day</b><br>' + \
            important_zones_text

        # Ranking of top 3 workers for the day by machine
        sorted_ranking = dict(sorted(
            self.quantity_over_time_per_machine.items(), key=lambda item: item[1], reverse=True))
        machine_id_ranking = ''
        cnt = 0
        for k, v in sorted_ranking.items():
            if cnt >= 3:
                break
            cnt += 1
            machine_id_ranking += f'<b>Nr.{cnt}</b> ID: {int(k)} moved {round(v*60,1)} t of mass moved per hour<br>'
        machine_id_ranking = f'<b> Top {cnt} workers of the day</b><br>' + \
            machine_id_ranking
        self.info_textbox.value = title+t_mass_moved + load_type_info + \
            important_zones_text+machine_id_ranking

    def get_optimal_k(self, X):
        """
        The function `get_optimal_k` uses Agglomerative Clustering to find the optimal number of
        clusters (k) for a given dataset by evaluating the differences in the scree plot.

        :param X: The parameter X is a numpy array that represents the dataset on which the clustering
        algorithm will be applied. It should have shape (n_samples, n_features), where n_samples is the
        number of data points and n_features is the number of features or dimensions of each data point
        :return: the optimal value of k, which is determined by evaluating the differences in the scree
        plot.
        """
        # Agglomerative Clustering for Scree plot
        y = []
        n_range = range(1, X.shape[0])
        for k in n_range:
            Hclustering = AgglomerativeClustering(n_clusters=k,
                                                  metric='euclidean',
                                                  linkage='ward',
                                                  compute_distances=True).fit(X)
            y.append(Hclustering.distances_)
        y = np.array(y[0])

        # find optimal k by evaluating diffs in the scree plot
        prev_val, opt_k = 0, 0
        prev_k = 0
        thres = 5e-3
        for k, val in zip(range(1, len(y)-1), y[::-1]):
            if abs(prev_val - val) < thres:
                opt_k = prev_k
                break
            prev_val = val
            prev_k = k

        return opt_k

    def dump_and_load_clustering(self):
        """
        The function `dump_and_load_clustering` performs clustering on load and dump locations based on
        machine trips and assigns labels to each location.
        """
        dump_latlons_and_quantity, load_latlons_and_quantity = [], []
        for machine_number in self.trips._machines.keys():
            temp_machine = self.trips._machines[machine_number]
            if temp_machine.machine_type == self.machine_type:
                for trip in temp_machine.trips:
                    load_latlons_and_quantity.append(
                        trip.load_latlon + (round(trip.quantity, 1), temp_machine.machine_id, trip.duration, trip.load))
                    dump_latlons_and_quantity.append(
                        trip.dump_latlon + (round(trip.quantity, 1), temp_machine.machine_id, trip.duration, trip.load))

        self.dump_latlons = np.array([[item[0], item[1]]
                                      for item in dump_latlons_and_quantity])
        self.load_latlons = np.array([[item[0], item[1]]
                                      for item in load_latlons_and_quantity])
        self.quantity = np.array([item[2]
                                  for item in dump_latlons_and_quantity])
        self.load_type = np.array([item[5]
                                  for item in dump_latlons_and_quantity])

        # make ranking of mass moved per time for each machine ID
        duration_id_quant = np.array(
            [(item[4], item[3], item[2]) for item in dump_latlons_and_quantity])
        self.quantity_over_time_per_machine = {}
        for _, id, _ in duration_id_quant:
            self.quantity_over_time_per_machine[id] = []
        for dur, id, quant in duration_id_quant:
            self.quantity_over_time_per_machine[id].append(quant/dur)
        for _, id, _ in duration_id_quant:
            self.quantity_over_time_per_machine[id] = round(
                np.average(self.quantity_over_time_per_machine[id]), 2)

        # Draw scree plot to find optimal K (can be extended to automaticly finding K based on gradient info)
        if self.k_dump is None:
            self.k_dump = self.get_optimal_k(self.dump_latlons)
        if self.k_load is None:
            self.k_load = self.get_optimal_k(self.load_latlons)

        # Find clusters of load and dump locations, input found optimal K for load and dump correspondingly
        ac2_dump = AgglomerativeClustering(
            n_clusters=self.k_dump).fit(self.dump_latlons)
        ac2_load = AgglomerativeClustering(
            n_clusters=self.k_load).fit(self.load_latlons)
        self.load_labels = ac2_load.labels_
        self.dump_labels = ac2_dump.labels_

    def find_center_latlon(self, latlons):
        """
        The function finds the center latitude and longitude from a list of latitude and longitude
        coordinates.

        :param latlons: The parameter "latlons" is a list of latitude and longitude coordinates. Each
        coordinate is represented as a tuple with two elements: the latitude and longitude values
        :return: a tuple containing the mean latitude and longitude values of the input latlons array.
        """
        return tuple(np.mean(latlons, axis=0))

    def find_quantity_moved_per_region(self):
        """
        The function `find_quantity_moved_per_region` calculates the quantity moved per region and
        stores the results in a dictionary.
        """
        self.quantity_by_region = {'dump': {}, 'load': {}, 'load_to_dump': {}}

        # initialize the dictionary per load and dump region
        for dump_label, load_label in zip(self.dump_labels, self.load_labels):
            self.quantity_by_region['dump'][dump_label] = 0
            self.quantity_by_region['load'][load_label] = 0
            self.quantity_by_region['load_to_dump']['L' +
                                                    str(load_label)+'->'+'D'+str(dump_label)] = 0
        # accumulated quantities over load and dump regions
        for dump_label, load_label, quant in zip(self.dump_labels, self.load_labels, self.quantity):
            # print(dump_label, load_label, quant)
            self.quantity_by_region['dump'][dump_label] += quant
            self.quantity_by_region['load'][load_label] += quant
            self.quantity_by_region['load_to_dump']['L' +
                                                    str(load_label)+'->'+'D'+str(dump_label)] += quant

    def populate_map(self, dump_or_load):
        """
        The `populate_map` function populates a map with polygons and markers based on given latitude
        and longitude coordinates, labels, and quantities.

        :param dump_or_load: The parameter `dump_or_load` is a string that specifies whether the
        function should populate the map with dump or load data. It can have two possible values: 'dump'
        or 'load'
        """
        poly_group = LayerGroup()
        marker_group = LayerGroup()
        if dump_or_load == 'dump':
            latlons = self.dump_latlons
            labels = self.dump_labels
        else:
            latlons = self.load_latlons
            labels = self.load_labels

        n_clusters = len(np.unique(labels))

        for l in range(n_clusters):
            current_latlons = np.array([(lat, lon) for (lat, lon),
                                        label in zip(latlons, labels) if label == l])
            current_quantity = self.quantity_by_region[dump_or_load][l]
            # needs 3 points to create hull
            if current_latlons.shape[0] >= 3:
                current_hull = ConvexHull(current_latlons)
                current_latlons = [(e[0], e[1])
                                   for e in current_latlons[current_hull.vertices, :]]
            else:
                current_latlons = [(e[0], e[1]) for e in current_latlons]

            # plot hull or line with cumulative mass moved per region html popup when clicked
            color = 'blue' if dump_or_load == 'dump' else 'red'
            poly = Polygon(locations=current_latlons,
                           color=color, smooth_factor=20)

            # Plot an indication marker at each dump and load zone
            marker_pretext = 'D' if dump_or_load == 'dump' else 'L'
            marker_text = marker_pretext + str(l)
            icon = DivIcon(html=marker_text, icon_size=[16, 16])
            marker = Marker(location=self.find_center_latlon(
                current_latlons), icon=icon)
            # add popup for total mass loaded or dumped
            message = HTML()
            if dump_or_load == 'dump':
                message.value = f'Total mass dumped here: {round(current_quantity, 1)} t'
            else:
                message.value = f'Total mass loaded here: {round(current_quantity, 1)} t'
            marker.popup = message
            marker_group.add_layer(marker)
            poly_group.add_layer(poly)
        self.m.add_layer(poly_group)
        self.m.add_layer(marker_group)

        self.overlays.extend([poly_group, marker_group])
