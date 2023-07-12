import pandas as pd

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
    trip_df = pd.read_csv(f'data/GPSData/trips/' +
                            file_name, index_col=None, header=0)
    info_df = pd.read_csv(f'data/GPSData/tripsInfo/' +
                            file_name, index_col=None, header=0)


    time_stamp_format1 = pd.to_datetime(trip_df["Timestamp"], errors="coerce")
    time_stamp_format2 = pd.to_datetime(trip_df["Timestamp"], errors="coerce", format="%Y-%m-%d %H:%M:%S%z")
    mask = time_stamp_format1.isna()

    time_stamp_format1[mask] = time_stamp_format2[mask]

    trip_df["Timestamp"] = time_stamp_format1

    def index_machine_name(row: pd.Series) -> pd.Series:
        print("Indexing With DumperMachineName", -MACHINE_NAMES.index(row["DumperMachineName"]))
        row["DumperMachineNumber"] = -MACHINE_NAMES.index(row["DumperMachineName"])
        return row

    info_df[info_df['DumperMachineNumber'].isna()] = info_df[info_df['DumperMachineNumber'].isna()].apply(index_machine_name)
    if 'DumperMachineName' not in info_df:
        info_df['DumperMachineName'] = None


    return info_df, trip_df
