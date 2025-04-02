import argparse
import pandas as pd
import mlflow

all_steps = ["data_import", "model_build"]


def run(args):
    data_url = args.data_url
    steps = args.steps
    active_steps = steps.split(",")
    print(active_steps)

    if "data_import" in active_steps:
        _ = mlflow.run(
            f"data_import/",
            "main",
            parameters={
                "data_url": data_url,
            },
        )

    if "model_build" in active_steps:
        _ = mlflow.run(f"model_build/", "main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=str, required=True)
    parser.add_argument("--data_url", type=str, required=True)
    args = parser.parse_args()

    run(args)