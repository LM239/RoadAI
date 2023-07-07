import pandas as pd
from datetime import date
from schemas import Machine, Trip, Position
import numpy as np


class TripsLoader:
    """
        Loads trips for one day
    """

    def __init__(self, datestring: str) -> None:
        self._machines: list[Machine] = []

        file_name = f'{datestring}.csv'

        trip_df = pd.read_csv(f'data/GPSData/trips/' +
                              file_name, index_col=None, header=0)
        info_df = pd.read_csv(f'data/GPSData/tripsInfo/' +
                              file_name, index_col=None, header=0)

        trip_df["Timestamp"] = pd.to_datetime(
            trip_df["Timestamp"], errors="coerce")
        info_df = info_df[~info_df['DumperMachineNumber'].isna()]

        if 'DumperMachineName' not in info_df:
            info_df['DumperMachineName'] = None

        grouped_df = trip_df.groupby("TripLogId")
        unique = set(trip_df["TripLogId"].unique())

        def get_trips(id: str):
            return {'route': grouped_df.get_group(id).drop("TripLogId", axis=1).values} if id in unique else None

        combined_df = info_df.copy()
        combined_df['route'] = combined_df.apply(
            lambda row: pd.Series(get_trips(row["TripLogId"])), axis=1)

        machine_groups = combined_df.groupby("DumperMachineNumber")

        for machine_id in info_df['DumperMachineNumber'].unique():
            machine_df: pd.DataFrame = machine_groups.get_group(
                machine_id).copy().reset_index(drop=True)
            if len(machine_df) == 0:
                continue

            machine_type, machine_name = (
                machine_df.at[0, 'MachineType'], machine_df.at[0, 'DumperMachineName'])
            machine_df = machine_df.drop(
                ["MachineType", "DumperMachineName"], axis=1)

            trips = []
            for index, row in machine_df.iterrows():
                row.to_dict()
                positions = [Position(timestamp=timestamp, lat=lat, lon=lon, uncertainty=uncertainty)
                             for timestamp, lat, lon, uncertainty in row["route"]]
                trip = Trip(
                    load=row['MassTypeMaterial'],
                    quantity=row['Quantity'],
                    positions=positions,
                )
                trips.append(trip)

            machine = Machine(machine_type=machine_type,
                              machine_id=machine_id, machine_name=machine_name, trips=trips)
            self._machines.append(machine)


if __name__ == "__main__":
    loader = TripsLoader('03-07-2022')
    ls = np.array(
        [trip.length for machine in loader._machines for trip in machine.trips])
    print(np.min(ls), np.max(ls), np.median(ls), np.mean(ls), np.std(ls))
