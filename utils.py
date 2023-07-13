import pandas as pd
from pathlib import Path

MACHINE_NAMES = ['Volvo A45 (4060) 12324060', 'A45 FS (3834) 12323834',
                 'Scania R580 (AJ91132)', 'Mercedes Arocs (DR67820)',
                 'Scania R590 (AJ94392) AJ94392 ', 'Mercedes (SD89781) 2763',
                 'Scania R580 AJ91826', 'Scania R590 AJ94391',
                 'Scania R580 (PD 70495)', 'Scania R580 (AJ90818)',
                 'SCANIA R 520 (PD 69848)', 'Mercedes Arocs (SD95898) 2902',
                 'Volvo A45G FS (3834) 12323834', 'Cat 745 B ( 1484 ) 12321484',
                 'SCANIA R490 8x4 4AKSLET 2505', 'Scania 590 (AJ94391)',
                 'Scania R540 AJ94080', 'Scania R 580 (PD 69849)', 'PD 69848']


def load_csv_from_date(file_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    File_name is a string on the formate 'YYYY-MM-DD'
    Loads the info and trip csv file for teh given date.
    Timestamp columns are transofmred to datetimes, MachineNumber and MachineName are combined as MachineID
    """
    trip_df = pd.read_csv(f'data/GPSData/trips/' +
                          file_name, index_col=None, header=0)
    info_df = pd.read_csv(f'data/GPSData/tripsInfo/' +
                          file_name, index_col=None, header=0)

    time_stamp_format1 = pd.to_datetime(trip_df["Timestamp"], errors="coerce")
    mask = time_stamp_format1.isna()
    time_stamp_format2 = pd.to_datetime(
        trip_df["Timestamp"][mask], errors="coerce", format="%Y-%m-%d %H:%M:%S%z")
    time_stamp_format1[mask] = time_stamp_format2

    trip_df["Timestamp"] = time_stamp_format1

    def index_machine_name(row: pd.Series) -> pd.Series:
        print("Indexing With DumperMachineName", -
              MACHINE_NAMES.index(row["DumperMachineName"]))
        row["DumperMachineNumber"] = - \
            MACHINE_NAMES.index(row["DumperMachineName"])
        return row

    info_df[info_df['DumperMachineNumber'].isna(
    )] = info_df[info_df['DumperMachineNumber'].isna()].apply(index_machine_name)
    if 'DumperMachineName' not in info_df:
        info_df['DumperMachineName'] = None

    return info_df, trip_df


def concat_df(files: list[Path]):
    """
    Loads the given files as dataframes, concatenetes the results, and resets the index.
    """
    li = []

    for filename in files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame.reset_index(drop=True)
