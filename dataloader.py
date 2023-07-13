import pandas as pd
from datetime import date
from schemas import Machine, Trip, Position
import numpy as np
from utils import load_csv_from_date
from typing import Literal


class TripsLoader:
    """
        Loads trips for one day
    """

    def __init__(self, datestring: str) -> None:
        self._machines: dict[int, Machine] = {}

        info_df, trip_df = load_csv_from_date(f'{datestring}.csv')

        grouped_df = trip_df.groupby("TripLogId")
        unique = set(trip_df["TripLogId"].unique())

        def get_trips(id: str):
            # return all the values assosciated with trip_log_id id ( if it exists in trip_df)
            return {'route': grouped_df.get_group(id).drop("TripLogId", axis=1).values} if id in unique else None

        combined_df = info_df.copy()
        # Combines the trip_info row, with all logged positions from trip_df
        combined_df['route'] = combined_df.apply(
            lambda row: pd.Series(get_trips(row["TripLogId"])), axis=1)

        machine_groups = combined_df.groupby("DumperMachineNumber")

        # Use groupby on machine to start populating machine basemodels
        for machine_id in info_df['DumperMachineNumber'].unique():
            machine_df: pd.DataFrame = machine_groups.get_group(
                machine_id).reset_index(drop=True)
            if len(machine_df) == 0:
                continue

            # Extarct machine_name and type and remove columns
            machine_type, machine_name = (
                machine_df.at[0, 'MachineType'], machine_df.at[0, 'DumperMachineName'])
            machine_df = machine_df.drop(
                ["MachineType", "DumperMachineName"], axis=1)

            trips = []
            # Extract all trips for this machine
            for _, row in machine_df.iterrows():
                row.to_dict()
                # Extract all positions from the route for the given row/trip
                positions = [Position(timestamp=timestamp, lat=lat, lon=lon, uncertainty=uncertainty)
                             for timestamp, lat, lon, uncertainty in row["route"]]
                trip = Trip(
                    trip_id=row["TripLogId"],
                    load=row['MassTypeMaterial'],
                    quantity=row['Quantity'],
                    dump_latlon=(row["DumpLatitude"], row["DumpLongitude"]),
                    load_latlon=(row["LoadLatitude"], row["LoadLongitude"]),
                    positions=positions,
                )
                trips.append(trip)

            # Save as a Machine model
            machine = Machine(machine_type=machine_type,
                              machine_id=machine_id, machine_name=machine_name, trips=trips)
            self._machines[machine_id] = machine

    @property
    def machines(self) -> dict[int, Machine]:
        return self._machines

    def sorted_machines(self, sort_by: Literal["trip_count", "trip_length", "total_quantity"], descending_order: bool = True) -> list[Machine]:
        """Return a sorted lsit of machine, either by the number of trips, length of trips
           or total quantity moved in the trips
        """

        # Define sorting keys
        def sort_by_count(machine: Machine):
            return len(machine.trips)

        def sort_by_length(machine: Machine):
            return sum(trip.length for trip in machine.trips)

        def sort_by_quantity(machine: Machine):
            return sum(trip.quantity for trip in machine.trips)

        # Select sorting key
        key = (sort_by_count if sort_by == "trip_count" else
               (sort_by_length if sort_by == "trip_length" else sort_by_quantity))

        return sorted(self._machines.values(), key=key, reverse=not descending_order)

    def filter_machine_type(self, m_type: Literal["Dumper", "Truck", "Dumpbil"]) -> dict[int, Machine]:
        """Return a list of machines that match the given machine type
        """
        def filter_by_type(pair: tuple[int, Machine]):
            _, machine = pair
            return machine.machine_type == m_type
        return dict(filter(filter_by_type, self._machines.values()))

# if __name__ == "__main__":
    # loader = TripsLoader('03-07-2022')
    # ls = np.array(
    #     [trip.length for machine in loader._machines for trip in machine.trips])
    # print(np.min(ls), np.max(ls), np.median(ls), np.mean(ls), np.std(ls))
