from functools import cached_property
from typing import Literal
from pydantic import BaseModel, computed_field
from datetime import datetime
import geopy.distance
from collections import defaultdict


class Position(BaseModel):
    lat: float
    lon: float
    uncertainty: float
    timestamp: datetime

    # easting: float
    # northing: float


class Trip(BaseModel):
    trip_id: str
    load: Literal['Stone', 'Equipment', 'Soil', '4']
    quantity: float
    positions: list[Position]
    dump_latlon: tuple[float, float]
    load_latlon: tuple[float, float]

    @computed_field
    @cached_property
    def latlons(self) -> list[tuple[float, float]]:
        # List of latlons
        return [(pos.lat, pos.lon) for pos in self.positions]

    @computed_field
    @cached_property
    def length(self) -> float:
        # Distacnce in kilometers
        return geopy.distance.geodesic(*self.latlons).km

    @computed_field
    @cached_property
    def duration(self) -> float:
        # Duration in minutes
        return (self.end_date - self.start_date).total_seconds() / 60.0

    @property
    def start_date(self) -> datetime:
        return self.positions[0].timestamp

    @property
    def end_date(self) -> datetime:
        return self.positions[-1].timestamp


class Machine(BaseModel):
    machine_type: Literal['Truck', 'Dumper', 'Tippbil']
    machine_id: int | str  # If looking at 2023 data needs to be string
    machine_name: str | None
    trips: list[Trip]

    @computed_field
    @cached_property
    def trips_dict(self) -> dict[Literal['Stone', 'Equipment', 'Soil', '4'], list[Trip]]:
        # Returns a dictionary of trips with key as the load type used
        partitions = defaultdict(lambda: [])
        set(partitions[trip.load].append(trip) for trip in self.trips)
        return partitions

    @computed_field
    @cached_property
    def total_length(self) -> float:
        # The combined length of all trips (in kilometers)
        return sum(trip.length for trip in self.trips)

    @computed_field
    @cached_property
    def total_duration(self) -> float:
        # The combined duration (in minutes) of each trip
        return sum(trip.duration for trip in self.trips)

    @computed_field
    @cached_property
    def total_quantity(self) -> float:
        # The combined quantity of each trip
        return sum(trip.quantity for trip in self.trips)
    
    @computed_field
    @cached_property
    def all_positions(self) -> list[Position]:
        all_pos = [trip.positions for trip in self.trips]
        all_pos = [item for sublist in all_pos for item in sublist]
        return all_pos

    @computed_field
    @cached_property
    def all_loads(self) -> list[Position]:
        temp_load : list[Position] = []
        for trip in self.trips:
            temp_load.append(Position(lat=trip.load_latlon[0],
                                            lon=trip.load_latlon[1], 
                                            uncertainty=0.0,
                                            timestamp=trip.positions[0].timestamp))
        return temp_load
    
    @computed_field
    @cached_property
    def all_dumps(self) -> list[Position]:
        temp_dump : list[Position] = []
        for trip in self.trips:
            temp_dump.append(Position(lat=trip.dump_latlon[0],
                                            lon=trip.dump_latlon[1], 
                                            uncertainty=0.0,
                                            timestamp=trip.positions[0].timestamp))
        return temp_dump
    
    @computed_field
    @cached_property
    def list_of_idle_times(self) -> list[list[Position]]:
        list_of_idle_times : list[list[Position]] = []
        temp_idle_list: list[Position] = []
        # We start processing. Are going to iterate over all positions, from first to last
        for i in range(1, len(self.all_positions[1:])):
            current_pos = self.all_positions[i]
            prev_pos = self.all_positions[i - 1]
            current_time = current_pos.timestamp

            # Seconds passed since last timestamp
            seconds_gone = (current_pos.timestamp -
                            prev_pos.timestamp).total_seconds()

            if seconds_gone > 0:
                # Meters driven since last timestamp
                meters_driven = geopy.distance.geodesic(
                    (current_pos.lat, current_pos.lon), (prev_pos.lat, prev_pos.lon)
                ).m

                # Speed during last timestamp
                speed_kmh = (meters_driven / seconds_gone) * 3.6

                if speed_kmh < 5:  # Speed limit - can be changed as user want
                    temp_idle_list.append(
                        Position(lat=current_pos.lat,
                                lon=current_pos.lon,
                                uncertainty=0,
                                timestamp=current_time)
                    )
                    if (
                        i == len(self.all_positions[1:]) - 1
                    ):  # i.e. last iteration
                        list_of_idle_times.append(temp_idle_list)
                else:
                    if len(temp_idle_list) > 0:
                        list_of_idle_times.append(temp_idle_list)
                        temp_idle_list = []  # Re-initialize
        
        return list_of_idle_times

    @computed_field
    @cached_property
    def total_idle_seconds(self) -> float:
        return sum(
                    [
                        (item[-1].timestamp - item[0].timestamp).total_seconds()
                        for item in self.list_of_idle_times
                    ]
                )


    @property
    def soil_trips(self):
        return self.trips_dict["Soil"]

    @property
    def equipment_trips(self) -> list[Trip]:
        return self.trips_dict["Equipment"]

    @property
    def stone_trips(self) -> list[Trip]:
        return self.trips_dict['Stone']

    @property
    def four_trips(self) -> list[Trip]:
        return self.trips_dict['4']
