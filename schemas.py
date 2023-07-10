from typing import Literal
from pydantic import BaseModel
from datetime import datetime
import geopy.distance


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

    @property
    def latlons(self) -> list[tuple[float, float]]:
        return [(pos.lat, pos.lon) for pos in self.positions]

    @property
    def length(self) -> float:
        return geopy.distance.geodesic(*self.latlons).km

    @property
    def start_date(self) -> datetime:
        return datetime.now()

    @property
    def end_date(self) -> datetime:
        return datetime.now()


class Machine(BaseModel):
    machine_type: Literal['Truck', 'Dumper', 'Tippbil']
    machine_id: int
    machine_name: str | None
    trips: list[Trip]
