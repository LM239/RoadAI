from sklearn.model_selection import train_test_split
import pandas as pd


def load_training_csv_files(
    name_of_data_set: str,
) -> pd.DataFrame:
    """
    Loads available training data
    """
    return pd.read_csv(
        f"data/ml_model_data/training_data/{name_of_data_set}", delimiter=","
    )


def split_data_into_training_and_testing(
    name_of_data_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and testing(unseen data)
    """
    data_set = load_training_csv_files(name_of_data_set)
    X, y = (
        data_set.drop(["MachineID", "output_labels", "DateTime"], axis=1),
        data_set["output_labels"],
    )
    # delete preds if they exist
    # _delete_pred_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40
    )
    return (X_train, X_test, y_train, y_test)


def if_duplicates():
    df = load_training_csv_files("1_day_all_trucks.csv")
    duplicates = df.duplicated()
    duplicated_rows = df[duplicates]
    return duplicated_rows


if __name__ == "__main__":
    print(if_duplicates())
