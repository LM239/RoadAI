import ipyleaflet as L
import numpy as np
from dataloader import TripsLoader
from ipywidgets import Layout, HTML, SelectionSlider, IntSlider, Output, VBox, HBox
from scipy.spatial import ConvexHull
from sklearn.cluster import AgglomerativeClustering


class InteractiveMap:
    def __init__(self, trips: TripsLoader) -> None:
        self.trips: TripsLoader = trips
        self.overlays: list = []
        self.k_load = None
        self.k_dump = None
        self.machine_type: str = 'Truck'
        self.day: str = trips.day

        self.m: L.Map = L.Map(layout=Layout(
            width='60%', height='500px'), center=(59.95, 10.35))

        self.k_dump_slider: IntSlider = IntSlider(min=2,
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
        self.k_load_slider = IntSlider(min=2,
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

        self.machine_type_slider = SelectionSlider(options=['Truck', 'Dumper'],
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

    def plot_interactive_map(self):
        quantity_by_region, quantity_over_time_per_machine = self.plot_map()
        self.info_textbox.value = self.update_textbox(
            quantity_by_region, quantity_over_time_per_machine)
        vbox_layout = VBox([self.k_dump_slider, self.k_load_slider,
                           self.machine_type_slider, self.info_textbox])
        hbox_layout = HBox([self.m, vbox_layout])
        display(hbox_layout)

    def update_k_dump(self, change):
        with self.out:
            self.k_dump = change['new']
            quantity_by_region, quantity_over_time_per_machine = self.plot_map()
            self.info_textbox.value = self.update_textbox(
                quantity_by_region, quantity_over_time_per_machine)

    def update_k_load(self, change):
        with self.out:
            self.k_load = change['new']
            quantity_by_region, quantity_over_time_per_machine = self.plot_map()
            self.info_textbox.value = self.update_textbox(
                quantity_by_region, quantity_over_time_per_machine)

    def update_machine_type(self, change):
        with self.out:
            self.machine_type = change['new']
            quantity_by_region, quantity_over_time_per_machine = self.plot_map()
            self.info_textbox.value = self.update_textbox(
                quantity_by_region, quantity_over_time_per_machine)

    def scree_plot(self, X):
        # Agglomerative Clustering for Scree plot
        y = []
        n_range = range(1, X.shape[0])
        for k in n_range:
            Hclustering = AgglomerativeClustering(
                n_clusters=k, metric='euclidean', linkage='ward', compute_distances=True).fit(X)
            y.append(Hclustering.distances_)
        y = np.array(y[0])

        # find optimal k by evaluating diffs in the scree plot
        prev_val, opt_k = 0, 0
        prev_k = 0
        thres = 0.01
        for k, val in zip(range(1, len(y)-1), y[::-1]):
            if abs(prev_val - val) < thres:
                opt_k = prev_k
                break
            prev_val = val
            prev_k = k

        return opt_k

    def dump_and_load_clustering(self):
        dump_latlons_and_quantity, load_latlons_and_quantity = [], []
        for machine_number in self.trips._machines.keys():
            temp_machine = self.trips._machines[machine_number]
            if temp_machine.machine_type == self.machine_type:
                for trip in temp_machine.trips:
                    load_latlons_and_quantity.append(
                        trip.load_latlon + (round(trip.quantity, 1), temp_machine.machine_id, trip.duration))
                    dump_latlons_and_quantity.append(
                        trip.dump_latlon + (round(trip.quantity, 1), temp_machine.machine_id, trip.duration))
        dump_latlons = np.array([[item[0], item[1]]
                                for item in dump_latlons_and_quantity])
        load_latlons = np.array([[item[0], item[1]]
                                for item in load_latlons_and_quantity])
        quantities = np.array([item[2] for item in dump_latlons_and_quantity])

        # make ranking of mass moved per time for each machine ID
        duration_id_quant = np.array(
            [(item[4], item[3], item[2]) for item in dump_latlons_and_quantity])
        quantity_over_time_per_machine = {}
        for _, id, _ in duration_id_quant:
            quantity_over_time_per_machine[id] = []
        for dur, id, quant in duration_id_quant:
            quantity_over_time_per_machine[id].append(quant/dur)
        for _, id, _ in duration_id_quant:
            quantity_over_time_per_machine[id] = round(
                np.average(quantity_over_time_per_machine[id]), 2)

        # Draw scree plot to find optimal K (can be extended to automaticly finding K based on gradient info)
        if self.k_dump == None:
            opt_k_dump = self.scree_plot(dump_latlons)
        else:
            opt_k_dump = self.k_dump
        if self.k_load == None:
            opt_k_load = self.scree_plot(load_latlons)
        else:
            opt_k_load = self.k_load
        # Find clusters of load and dump locations, input found optimal K for load and dump correspondingly
        ac2_dump = AgglomerativeClustering(
            n_clusters=opt_k_dump).fit(dump_latlons)
        ac2_load = AgglomerativeClustering(
            n_clusters=opt_k_load).fit(load_latlons)

        return load_latlons, dump_latlons, ac2_load.labels_, ac2_dump.labels_, quantities, quantity_over_time_per_machine

    def find_center_latlon(self, latlons):
        return tuple(np.mean(latlons, axis=0))

    def find_quantity_moved_per_region(self, quantity, dump_labels, load_labels):
        quantity_by_region = {'dump': {}, 'load': {}, 'load_to_dump': {}}

        # initialize the dictionary per load and dump region
        for dump_label, load_label in zip(dump_labels, load_labels):
            quantity_by_region['dump'][dump_label] = 0
            quantity_by_region['load'][load_label] = 0
            quantity_by_region['load_to_dump']['L' +
                                               str(load_label)+'->'+'D'+str(dump_label)] = 0
        # accumulated quantities over load and dump regions
        for dump_label, load_label, quant in zip(dump_labels, load_labels, quantity):
            # print(dump_label, load_label, quant)
            quantity_by_region['dump'][dump_label] += quant
            quantity_by_region['load'][load_label] += quant
            quantity_by_region['load_to_dump']['L' +
                                               str(load_label)+'->'+'D'+str(dump_label)] += quant
        return quantity_by_region

    def populate_map(self, latlons, labels, n_clusters, quantity_by_region, dump_or_load):
        poly_group = L.LayerGroup()
        marker_group = L.LayerGroup()
        for l in range(n_clusters):
            current_latlons = []
            for (lat, lon), label in zip(latlons, labels):
                if label == l:
                    current_latlons.append([lat, lon])
            current_latlons = np.array(current_latlons)
            current_quantity = quantity_by_region[dump_or_load][l]
            # needs 3 points to create hull
            if current_latlons.shape[0] >= 3:
                current_hull = ConvexHull(current_latlons)
                current_latlons = [(e[0], e[1])
                                   for e in current_latlons[current_hull.vertices, :]]
            else:
                current_latlons = [(e[0], e[1]) for e in current_latlons]

            # plot hull or line with cumulative mass moved per region html popup when clicked
            color = 'blue' if dump_or_load == 'dump' else 'red'
            poly = L.Polygon(locations=current_latlons,
                             color=color, smooth_factor=20)

            # Plot an indication marker at each dump and load zone
            marker_pretext = 'D' if dump_or_load == 'dump' else 'L'
            marker_text = marker_pretext + str(l)
            icon = L.DivIcon(html=marker_text, icon_size=[16, 16])
            marker = L.Marker(location=self.find_center_latlon(
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
        self.overlays.append(poly_group)
        self.overlays.append(marker_group)

    def plot_hulls_on_map(self, dump_latlons, load_latlons, dump_labels, load_labels, quantity):
        n_clusters_dump, n_clusters_load = len(
            np.unique(dump_labels)), len(np.unique(load_labels))
        quantity_by_region = self.find_quantity_moved_per_region(
            quantity, dump_labels, load_labels)

        #  populate map with dumps and loads
        self.populate_map(dump_latlons, dump_labels,
                          n_clusters_dump, quantity_by_region, 'dump')
        self.populate_map(load_latlons, load_labels,
                          n_clusters_load, quantity_by_region, 'load')
        return quantity_by_region

    def plot_map(self):
        # remove previous overlays
        for overlay in self.overlays:
            self.m.remove_layer(overlay)
        self.overlays.clear()

        # plot new polygons and markers
        load_latlons, dump_latlons, load_labels, dump_labels, quantity, quantity_over_time_per_machine = self.dump_and_load_clustering()
        quantity_by_region = self.plot_hulls_on_map(
            dump_latlons, load_latlons, dump_labels, load_labels, quantity)
        return quantity_by_region, quantity_over_time_per_machine

    def update_textbox(self, quantity_by_region, quantity_over_time_per_machine):
        title = f'<b>Day Overview, {self.day}</b><br>'

        # find total mass moved for the day
        t_mass = np.sum([e for e in quantity_by_region['dump'].values()])
        t_mass_moved = f'Total mass moved for the day by {self.machine_type}: {round(t_mass, 1)} t<br>'

        # list load to dump zones by importance (weight decending) [up to 5 instances]
        sorted_load_to_dump_zones = dict(sorted(
            quantity_by_region['load_to_dump'].items(), key=lambda item: item[1], reverse=True))
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
            quantity_over_time_per_machine.items(), key=lambda item: item[1], reverse=True))
        machine_id_ranking = ''
        cnt = 0
        for k, v in sorted_ranking.items():
            if cnt >= 3:
                break
            cnt += 1
            machine_id_ranking += f'<b>Nr.{cnt}</b> ID: {str(k)} moved {round(v*60,1)} t of mass moved per hour<br>'
        machine_id_ranking = f'<b> Top {cnt} workers of the day</b><br>' + \
            machine_id_ranking
        return title+t_mass_moved+important_zones_text+machine_id_ranking
