from functools import cached_property
from typing import Literal
from pydantic import BaseModel, computed_field
from datetime import datetime
import geopy.distance
from collections import defaultdict
import math

class Position(BaseModel):
    lat: float
    lon: float
    uncertainty: float
    timestamp: datetime

    # easting: float
    # northing: float


class Trip(BaseModel):
    load: Literal['Stone', 'Equipment', 'Soil', '4']
    quantity: float
    positions: list[Position]

    @computed_field
    @cached_property
    def latlons(self) -> list[tuple[float, float]]:
        return [(pos.lat, pos.lon) for pos in self.positions]
    
    @computed_field
    @cached_property
    def length(self) -> float:
        return geopy.distance.geodesic(*self.latlons).km
    
    @computed_field
    @cached_property
    def duration(self) -> float:
        return (self.end_date - self.start_date).total_seconds() / 60.0
    
    @property
    def start_date(self) -> datetime:
        return self.positions[0].timestamp

    @property
    def end_date(self) -> datetime:
        return self.positions[-1].timestamp


class Machine(BaseModel):
    machine_type: Literal['Truck', 'Dumper', 'Tippbil']
    machine_id: int
    machine_name: str | None
    trips: list[Trip]

    @computed_field
    @cached_property
    def trips_dict(self) -> dict[Literal['Stone', 'Equipment', 'Soil', '4'], list[Trip]]:
        partitions = defaultdict(lambda: [])
        set(partitions[trip.load].append(trip) for trip in self.trips)
        return partitions
    
    @computed_field
    @cached_property
    def total_length(self) -> float:
        return sum(trip.length for trip in self.trips)

    @computed_field
    @cached_property
    def total_duration(self) -> float:
        return sum(trip.duration for trip in self.trips)

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
