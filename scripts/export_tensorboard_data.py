import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import click
from pathlib import Path

# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

@click.command()
@click.option('--path', default=None, help='Path to the TensorBoard log directory.')
def export_tensorboard_data(path):
    # Use pathlib to determine the latest version directory
    if path is None:
        log_dir = Path('lightning_logs')
        # Find the latest version directory
        if log_dir.exists() and log_dir.is_dir():
            version_dirs = sorted(log_dir.glob('version_*'), key=lambda d: d.stat().st_mtime, reverse=True)
            if version_dirs:
                path = version_dirs[0]
            else:
                raise ValueError("No version directories found in 'lightning_logs'.")
        else:
            raise ValueError("'lightning_logs' directory does not exist.")

    df = tflog2pandas(str(path))
    df.to_csv("output.csv")

if __name__ == '__main__':
    export_tensorboard_data()