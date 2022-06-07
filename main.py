# Sound field reconstruction in rooms: inpainting meets superresolution - 17.12.2019
# Main.py

import argparse
import training
import inference
import evaluation


def main():
    """Reads command line arguments and starts either training or evaluation."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="", help="JSON-formatted file with configuration parameters"
    )
    parser.add_argument(
        "--mode",
        default="",
        help="The operational mode - train|sim-eval|real-eval|visualize",
    )
    cla = parser.parse_args()
    if cla.config == "":
        print(
            "Error: --config flag is required. Enter the path to a configuration file."
        )

    if cla.mode == "sim-eval":
        inference.simulated_data_evaluation(cla.config)
    elif cla.mode == "real-eval":
        inference.real_data_evaluation(cla.config)
    elif cla.mode == "ssim-nmse":
        evaluation.evaluate_general_ssim_nmse(cla.config)
        evaluation.compare_soundfields(cla.config)
    elif cla.mode == "train":
        training.train(cla.config)
    elif cla.mode == "visualize-real-room":
        inference.visualize_real(cla.config)
    elif cla.mode == "predict-sim":
        inference.predict_soundfield(cla.config)
    
    else:
        print(
            "Error: invalid operational mode - options: train, sim-eval, real-eval, visualize-real-room, ssim-nmse or predict-sim"
        )


if __name__ == "__main__":
    main()
