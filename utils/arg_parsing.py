import argparse


def parse_model_arguments():
    valid_model_versions = ["2016", "2023"]

    parser = argparse.ArgumentParser(description="Run model with specific version.")
    parser.add_argument(
        "--model_version", type=str, help="Version of the model to run", required=True
    )
    args = parser.parse_args()

    # Ensure that args.model_version exists and is stripped once
    model_version = args.model_version.strip() if args.model_version else None

    # First, check both for None and empty strings
    if not model_version:
        parser.error("The --model_version argument must not be empty.")

    # Then, check if the model version is valid
    if model_version not in valid_model_versions:
        valid_versions_str = ", ".join(valid_model_versions)
        parser.error(
            f"The --model_version argument must be for an existing model configuration: {valid_versions_str}."
        )

    return args
