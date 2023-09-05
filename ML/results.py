import pandas as pd
import matplotlib.pyplot as plt


def convert_to_0_and_1s(df: pd.DataFrame, output_label: str):
    """
    Returns df with only output_label in output_labels column
    """
    fitered_df = df[df["output_labels"] == output_label]
    return fitered_df


def get_probabilities(ndays: int, merged_timestamps: bool) -> None:
    """
    Calculates probabilities by using the complete test prediciton data set

    Get probabilities on form P(pred | event):

        P(pred_load | load)
        P(pred_driving | load)
        P(pred_dump | load)

        P(pred_load | driving)
        P(pred_driving | driving)
        P(pred_dump | driving)

        P(pred_load | dump)
        P(pred_driving | dump)
        P(pred_dump | dump)

    The probabilities of each bulk above add to one (100%)
    """
    pred_dict = {}
    df_preds = pd.read_csv(
        "data/ml_model_data/preds/preds_load_dump_testing.csv",
        sep=",",
        usecols=["output_labels", "proba_Driving", "proba_Dump", "proba_Load"],
    )
    events = ["Load", "Driving", "Dump"]
    pred_events = ["proba_Load", "proba_Driving", "proba_Dump"]
    for event in events:
        for pred_event in pred_events:
            filtered_df = df_preds[df_preds["output_labels"] == event]
            pred_dict[f"{pred_event} | {event} "] = filtered_df[pred_event].mean()

    pd.DataFrame(
        {"Condition": list(pred_dict.keys()), "Probabilities": list(pred_dict.values())}
    ).to_csv(
        f"data/ml_model_data/preds/probabilities{ndays}_merged_timestamps={merged_timestamps}.csv",
        index=False,
        sep=",",
    )


# def plot_subfigure(
#     ax, time: pd.DataFrame, proba_label: pd.DataFrame, label: str
# ) -> None:
#     ax.plot(time, proba_label, label=label)


def plot_visual_for_truck_on_single_day(
    merged_timestamps: bool,
    machine_ids: list[str | int] = [1],
    datestrings: list[str] = ["2022-03-07", "2022-03-08"],
):
    """
    Plot 3 by 1 subplot for each machine for each day

    Args:
        MachineID: id of machine, e.g. 20. String if 2023 data
        datestring: yyyy-mm-dd
    """

    df_preds = pd.read_csv(
        "data/ml_model_data/preds/preds_load_dump_testing.csv", sep=","
    )
    for machine_id in machine_ids:
        for datestring in datestrings:
            fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharey=False)
            # they share x axis so set axis only for one of the elements
            fig.suptitle(
                f"Load and dump cycle for machine with id: {machine_id}. Date: {datestring}"
            )

            df_chosen_machine = df_preds[
                (df_preds["MachineID"] == machine_id)
                & (
                    df_preds[
                        "DateTime_min" if merged_timestamps else "DateTime"
                    ].str.startswith(datestring)
                )
            ]

            time = pd.to_datetime(
                df_chosen_machine["DateTime_min" if merged_timestamps else "DateTime"]
            )

            for label, label2, color, ax in zip(
                ["Load", "Dump", "Driving"],
                ["proba_Load", "proba_Dump", "proba_Driving"],
                ["tab:red", "tab:blue", "tab:green"],
                axs,
            ):
                mask = df_chosen_machine["output_labels"] == label
                filtered_proba_label = df_chosen_machine.loc[mask, label2]
                filtered_time = time[mask]
                ax.scatter(
                    filtered_time,
                    [1] * len(filtered_time),
                    label=f"True {label}",
                    s=10,
                    marker="x",
                    color="k",
                )
                ax.scatter(
                    filtered_time,
                    filtered_proba_label,
                    label=f"Predicted probability {label}",
                    s=4,
                    marker="o",
                    color=color,
                )
                ax.set_ylim(-0.1, 1.1)
                ax.set_xlabel("Time")
                ax.legend()
            fig.tight_layout()

            fig.savefig(f"data/ml_model_data/pngs/probability_outputs_{datestring}.png")
            break


if __name__ == "__main__":
    get_probabilities(1, merged_timestamps=False)
    plot_visual_for_truck_on_single_day(merged_timestamps=False)
    pass
