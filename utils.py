import pandas as pd

def load_csv_from_date(file_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    trip_df = pd.read_csv(f'data/GPSData/trips/' +
                            file_name, index_col=None, header=0)
    info_df = pd.read_csv(f'data/GPSData/tripsInfo/' +
                            file_name, index_col=None, header=0)

    trip_df["Timestamp"] = pd.to_datetime(
        trip_df["Timestamp"], errors="coerce")
    info_df = info_df[~info_df['DumperMachineNumber'].isna()]

    if 'DumperMachineName' not in info_df:
        info_df['DumperMachineName'] = None


    return info_df, trip_df
