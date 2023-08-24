from sklearn.model_selection import train_test_split
import pandas as pd

DATA_SETS = ["2_days_1_vehicle.csv", "1_day_all_trucks.csv"]


def data_set_to_consider() -> str:
    return DATA_SETS[0]


def load_training_csv_files(
    name_of_data_set: str = data_set_to_consider(),
) -> pd.DataFrame:
    """
    Loads available training data
    """
    return pd.read_csv(
        f"data/ml_model_data/training_data/{name_of_data_set}", delimiter=","
    )


def split_data_into_training_and_testing(
    all_data: pd.DataFrame = load_training_csv_files(),
) -> tuple[pd.DataFrame]:
    """
    Splits data into training and testing(unseen data)
    """
    X, y = (
        all_data.drop(["Load", "Dump", "DateTime", "Time_from_start"], axis=1),
        all_data[["Load", "Dump"]],
    )
    # delete preds if they exist
    # _delete_pred_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40
    )
    return (X_train, X_test, y_train, y_test)  # type: ignore
