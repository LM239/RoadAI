from typing import Literal
from pydantic import BaseModel
from datetime import datetime


class Position:
    lat: float
    lon: float
    easting: float
    northing: float

class Trip(BaseModel):
    load: Literal['Stone', 'Equipment', 'Soil', '4']
    quantity: int
    start_date: datetime
    end_date: datetime
    positions: list[Position]

class Machine(BaseModel):
    type: Literal['Truck', 'Dumper', 'Tippbil']
    machine_id: int
    machine_name: str
    trips: list[Trip]
