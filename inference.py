# Sound field reconstruction in rooms: inpainting meets superresolution - 17.12.2019
# Inference.py

import os
import util.util as util
import sfun
import data
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_general_checkpoint_path(session_dir, number_checkpoint):
    """Returns the path of the most recent checkpoint in session_dir.

    Args:
    session_dir: string
    number_epoch: int

    Returns: string

    """
    checkpoints_path = os.path.join(session_dir, "checkpoints")
    if os.path.exists(checkpoints_path) and util.dir_contains_files(checkpoints_path):
        checkpoints = os.listdir(checkpoints_path)
        checkpoints.sort(
            key=lambda x: os.stat(os.path.join(checkpoints_path, x)).st_mtime
        )
        last_checkpoint = checkpoints[number_checkpoint]
        return os.path.join(checkpoints_path, last_checkpoint)
    else:
        return ""


def get_latest_checkpoint_path(session_dir):
    """Returns the path of the most recent checkpoint in session_dir.

    Args:
    session_dir: string

    Returns: string

    """
    checkpoints_path = os.path.join(session_dir, "checkpoints")
    if os.path.exists(checkpoints_path) and util.dir_contains_files(checkpoints_path):
        checkpoints = os.listdir(checkpoints_path)
        checkpoints.sort(
            key=lambda x: os.stat(os.path.join(checkpoints_path, x)).st_mtime
        )
        last_checkpoint = checkpoints[-1]
        return os.path.join(checkpoints_path, last_checkpoint)
    else:
        return ""


def get_results_dict():
    """Get the dictionary to save results.

    Returns: dict

    """
    return {
        "name": [],
        "num_file": [],
        "xDim": [],
        "yDim": [],
        "m2": [],
        "num_mics": [],
        "num_comb": [],
        "freq": [],
        "NMSE": [],
        "SSIM": [],
        "pattern": [],
        "p_real": [],
        "p_predicted": [],
        "p_previous": [],
    }


def get_test_filenames(test_path):
    """Get the .mat filenames given a folder path.

    Args:
    test_path: string

    Returns: string

    """
    # Get file from subfolders
    filenames = []
    for subfolder in os.listdir(test_path):
        subfolder_path = os.path.join(test_path, subfolder)
        filenames_list = [
            os.path.join(subfolder, filename)
            for filename in os.listdir(subfolder_path)
            if filename.endswith(".mat")
        ]
        filenames.extend(filenames_list)
    # filenames = [filename for filename in os.listdir(test_path) if filename.endswith('.mat')]
    return filenames


def reconstruct_soundfield(
    model,
    sf_sample,
    mask,
    factor,
    frequencies,
    filename,
    num_file,
    com_num,
    results_dict,
):
    """Reconstruct and evaluate sound field

    Args:
    model: keras model
    sf_sample: np.ndarray
    factor: int
    frequencies: list
    filename: string
    num_file: int
    com_num: int
    results_dict: dict



    Returns: dict

    """

    # Create one sample batch. Expand dims
    sf_sample = np.expand_dims(sf_sample, axis=0)
    sf_gt = copy.deepcopy(sf_sample)
    sf_sample[:, :, :, 0].shape
    mask = np.expand_dims(mask, axis=0)
    mask_gt = copy.deepcopy(mask)

    # preprocessing
    irregular_sf, mask = util.preprocessing(factor, sf_sample, mask)
    # predict sound field
    pred_sf = model.predict([irregular_sf, mask])

    # measured observations. To use in postprocessing
    measured_sf = util.downsampling(factor, copy.deepcopy(sf_gt))
    measured_sf = util.apply_mask(measured_sf, mask_gt)

    # compute csv fields
    split_filename = filename[:-4].split("_")
    pattern = np.where(mask_gt[0, :, :, 0].flatten() == 1)[0]
    num_mic = len(pattern)

    for freq_num, freq in enumerate(frequencies):
        # Postprocessing
        reconstructed_sf_slice = util.postprocessing(
            pred_sf, measured_sf, freq_num, pattern, factor
        )

        # Compute Metrics
        reconstructed_sf_slice = util.postprocessing(
            pred_sf, measured_sf, freq_num, pattern, factor
        )
        nmse = util.compute_NMSE(sf_gt[0, :, :, freq_num], reconstructed_sf_slice)

        data_range = sf_gt[0, :, :, freq_num].max() - sf_gt[0, :, :, freq_num].min()
        ssim = util.compute_SSIM(
            sf_gt[0, :, :, freq_num].astype("float32"),
            reconstructed_sf_slice,
            data_range,
        )

        average_pressure_real = util.compute_average_pressure(sf_gt[0, :, :, freq_num])
        average_pressure_predicted = util.compute_average_pressure(
            reconstructed_sf_slice
        )
        average_pressure_previous = util.compute_average_pressure(
            measured_sf[0, :, :, freq_num]
        )

        # store results
        results_dict["freq"].append(freq)
        results_dict["name"].append(filename[:-4])
        results_dict["xDim"].append(split_filename[2])
        results_dict["yDim"].append(split_filename[3])
        results_dict["m2"].append(split_filename[4])
        results_dict["num_mics"].append(num_mic)
        results_dict["num_comb"].append(com_num)
        results_dict["num_file"].append(num_file)
        results_dict["pattern"].append(pattern)
        results_dict["NMSE"].append(nmse)
        results_dict["SSIM"].append(ssim)
        results_dict["p_real"].append(average_pressure_real)
        results_dict["p_predicted"].append(average_pressure_predicted)
        results_dict["p_previous"].append(average_pressure_previous)

    return results_dict


def real_data_evaluation(config_path):
    """Evaluates a trained model on real data.

    Args:
    config_path: string

    """

    config = util.load_config(config_path)
    print("Loaded configuration from: %s" % config_path)

    session_dir = config_path[: config_path.rfind("/") + 1]

    checkpoint_path = get_latest_checkpoint_path(session_dir)
    if not checkpoint_path:
        print("Error: No checkpoint found in same directory as configuration file.")
        return

    model = sfun.SFUN(config, train_bn=False)

    predict_path = os.path.join(
        session_dir,
        "real_data_evaluation",
        "min_mics_"
        + str(config["evaluation"]["min_mics"])
        + "_max_mics_"
        + str(config["evaluation"]["max_mics"])
        + "_step_mics_"
        + str(config["evaluation"]["step_mics"]),
    )
    if not os.path.exists(predict_path):
        os.makedirs(predict_path)

    filepath = os.path.join(
        config["dataset"]["path"], "real_soundfields", "RoomB_soundfield.mat"
    )

    # Get Ground Truth
    soundfield_1 = util.load_RoomB_soundfield(filepath, 0)
    soundfield_2 = util.load_RoomB_soundfield(filepath, 1)

    frequencies = util.get_frequencies()

    results_dict = get_results_dict()

    print("\nEvaluating model in real sound fields...\n")

    for num_mics in range(
        config["evaluation"]["min_mics"],
        config["evaluation"]["max_mics"],
        config["evaluation"]["step_mics"],
    ):
        for source_num, source in enumerate(["source_1", "source_2"]):

            mask_generator = data.MaskGenerator(
                config["dataset"]["xSamples"] // config["dataset"]["factor"],
                config["dataset"]["ySamples"] // config["dataset"]["factor"],
                len(frequencies),
                num_mics=num_mics,
                rand_seed=3,
            )

            for com_num in range(config["evaluation"]["num_comb"]):
                print(
                    "\twith "
                    + str(num_mics)
                    + " mics, pattern number "
                    + str(com_num)
                    + " and source position "
                    + str(source_num)
                )
                if source_num:
                    input_soundfield = copy.deepcopy(soundfield_2)
                else:
                    input_soundfield = copy.deepcopy(soundfield_1)

                mask = mask_generator.sample()
                filename = str(source_num) + "_d_4.159_6.459_26.862.mat"

                results_dict = reconstruct_soundfield(
                    model,
                    input_soundfield,
                    mask,
                    config["dataset"]["factor"],
                    frequencies,
                    filename,
                    source_num,
                    com_num,
                    results_dict,
                )

    print("\nWriting real room results...\n")
    util.write_results(
        os.path.join(predict_path, "results_RoomB_Dataset.csv"), results_dict
    )

    print("Analysing and plotting results...")
    util.analyze_and_plot_real_results(
        os.path.join(predict_path, "results_RoomB_Dataset.csv"), config
    )

    print("Evaluation completed!")


def simulated_data_evaluation(config_path):
    """Evaluates a trained model on simulated data.

    Args:
    config_path: string

    """

    config = util.load_config(config_path)
    print("Loaded configuration from: %s" % config_path)

    session_dir = config_path[: config_path.rfind("\\") + 1]

    # checkpoint_path = get_latest_checkpoint_path(session_dir)
    checkpoint_path = get_general_checkpoint_path(
        session_dir=session_dir, number_checkpoint=-1
    )
    if not checkpoint_path:  # Model weights are loaded when creating the model object
        print("Error: No checkpoint found in same directory as configuration file.")
        return

    model = sfun.SFUN(config, train_bn=False)

    evaluation_path = os.path.join(
        session_dir,
        "simulated_data_evaluation",
        "min_mics_"
        + str(config["evaluation"]["min_mics"])
        + "_max_mics_"
        + str(config["evaluation"]["max_mics"])
        + "_step_mics_"
        + str(config["evaluation"]["step_mics"]),
    )

    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)
    
    test_path = os.path.join(config["storage"]["path"],"datasets",config["dataset"]["name"],"simulated_soundfields", "test")
    filenames = get_test_filenames(test_path)

    frequencies = util.get_frequencies()
    for num_file, filename in enumerate(sorted(filenames)):
        print("\nEvaluating model in simulated room " + str(num_file) + "...\n")
        aux_sound = util.load_soundfield(os.path.join(test_path, filename), frequencies)
        results_dict = get_results_dict()

        for num_mics in range(
            config["evaluation"]["min_mics"],
            config["evaluation"]["max_mics"],
            config["evaluation"]["step_mics"],
        ):
            mask_generator = data.MaskGenerator(
                config["dataset"]["xSamples"] // config["dataset"]["factor"],
                config["dataset"]["ySamples"] // config["dataset"]["factor"],
                len(frequencies),
                num_mics=num_mics,
                rand_seed=3,
            )

            for com_num in range(config["evaluation"]["num_comb"]):
                soundfield_input = copy.deepcopy(aux_sound)

                mask = mask_generator.sample()

                print(
                    "\twith "
                    + str(num_mics)
                    + " mics and pattern number "
                    + str(com_num)
                )

                results_dict = reconstruct_soundfield(
                    model,
                    soundfield_input,
                    mask,
                    config["dataset"]["factor"],
                    frequencies,
                    filename,
                    num_file,
                    com_num,
                    results_dict,
                )

                com_num += 1

        filename = "results_file_number_" + str(num_file) + ".csv"
        print("\nWriting simulated room " + str(num_file) + " results...\n")
        util.write_results(os.path.join(evaluation_path, filename), results_dict)

    print("Analysing and plotting results...")

    util.analyze_and_plot_simulated_results(evaluation_path, config)
    
    history_file = "".join(["history_session_",str(config["training"]["session_id"]),".csv"])
    session_history_path = os.path.join(session_dir,history_file)
    plot_training_loss(history_path=session_history_path)

    print("Evaluation completed!")

def plot_training_loss(history_path:str):
    """
    Plots loss and PSNR from the csv file
    from "path"
    """

    df = pd.read_csv(history_path)
    current_session = history_path.split("\\")[-2]
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.plot(df["epoch"],df["loss"],"b",linewidth=2)
    plt.plot(df["epoch"],df["val_loss"],"k",linewidth=2)
    plt.grid()
    plt.legend(["loss","val_loss"],loc=5, prop={'size': 15})
    plt.title(f"Training Losses ({current_session})")
    plt.ylim([0,5])
    plt.xlabel("Epoch")
    
    plt.subplot(2,1,2)
    plt.plot(df["epoch"],df["PSNR"],"r",linewidth=2)
    plt.plot(df["epoch"],df["val_PSNR"],"m",linewidth=2)
    plt.grid()
    plt.legend(["PSNR","val_PSNR"],loc=5, prop={'size': 15})
    plt.title(f"Training PSNR ({current_session})")
    plt.ylim([10,20])
    plt.xlabel("Epoch")

    config_path = os.path.join(history_path[:history_path.rfind("\\")],"config.json")
    config = util.load_config(config_path)
    save_path = os.path.join(history_path[:history_path.rfind("\\")],
                                        "simulated_data_evaluation",
                                        "min_mics_"
                                        + str(config["evaluation"]["min_mics"])
                                        + "_max_mics_"
                                        + str(config["evaluation"]["max_mics"])
                                        + "_step_mics_"
                                        + str(config["evaluation"]["step_mics"])
                                    )

    # if not os.path.exists(individual_results_path):
    #     os.mkdir(individual_results_path)
    plt.savefig(os.path.join(save_path,"session_loss_history.png"))


def visualize(config_path):
    """Plot predictions of trained model on real data.

    Args:
    config_path: string

    """

    config = util.load_config(config_path)
    print("Loaded configuration from: %s" % config_path)

    frequencies = util.get_frequencies()

    session_dir = config_path[: config_path.rfind("\\") + 1]

    checkpoint_path = get_general_checkpoint_path(
        session_dir=session_dir, number_checkpoint=-1
    )
    if not checkpoint_path:
        print("Error: No checkpoint found in same directory as configuration file.")
        return

    model = sfun.SFUN(config, train_bn=False)

    visualization_path = os.path.join(session_dir,'visualization')
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)

    dataset_path = os.path.join(config["storage"]["path"],"datasets", config["dataset"]["name"])
    real_room_filepath =  os.path.join(dataset_path,"real_soundfields","RoomB_soundfield.mat")

    if not os.path.exists(real_room_filepath):
        
        os.mkdir("".join([dataset_path, "/real_soundfields"]))
        source_path = os.path.join(config["storage"]["path"],"datasets","real_soundfield_sample","RoomB_soundfield.mat")
        destination_path = os.path.join(dataset_path,"real_soundfields","RoomB_soundfield.mat")
        
        os.system(f"copy {source_path} {destination_path}")

    mask_generator = data.MaskGenerator(
        config["dataset"]["xSamples"] // config["dataset"]["factor"],
        config["dataset"]["ySamples"] // config["dataset"]["factor"],
        len(frequencies),
        num_mics=config["visualization"]["num_mics"],
    )

    # Get measured sound field

    sf_sample = util.load_RoomB_soundfield(
        real_room_filepath, config["visualization"]["source"]
    )
    sf_gt = np.expand_dims(copy.deepcopy(sf_sample), axis=0)
    initial_sf = np.expand_dims(sf_sample, axis=0)

    # Get mask samples
    mask = mask_generator.sample()
    mask = np.expand_dims(mask, axis=0)

    # Preprocessing
    irregular_sf, mask = util.preprocessing(
        config["dataset"]["factor"], initial_sf, mask
    )

    # Save range to allow colorbar
    util.save_pressure_range(sf_gt, visualization_path)
    print("\tPressure range saved")
    # Scale ground truth sound field
    sf_gt = util.scale(sf_gt)

    print("\nPlotting Ground Truth Sound Field Scaled...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            sf_gt[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Ground_Truth.png"),
        )

    print("\nPlotting Irregular Sound Field...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            irregular_sf[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Irregular_SF.png"),
        )

    print("\nPlotting Mask...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            mask[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Mask.png"),
        )

    pred_sf = model.predict([irregular_sf, mask])

    print("\nPlotting Predicted Sound Field...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            pred_sf[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Pred_SF.png"),
        )

def visualize_general(config_path):
    """Plot predictions of trained model on SIMULATED data.

    Args:
    config_path: string

    """

    config = util.load_config(config_path)
    print("Loaded configuration from: %s" % config_path)

    frequencies = util.get_frequencies()

    session_dir = config_path[: config_path.rfind("\\") + 1]

    checkpoint_path = get_general_checkpoint_path(
        session_dir=session_dir, number_checkpoint=-1
    )
    if not checkpoint_path:
        print("Error: No checkpoint found in same directory as configuration file.")
        return

    model = sfun.SFUN(config, train_bn=False)

    visualization_path = os.path.join(session_dir,'visualization')
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)

    dataset_path = os.path.join(config["storage"]["path"],"datasets", config["dataset"]["name"])
    visualized_room_dir = os.path.join(dataset_path,"visualized_soundfields")
    
    mask_generator = data.MaskGenerator(
        config["dataset"]["xSamples"] // config["dataset"]["factor"],
        config["dataset"]["ySamples"] // config["dataset"]["factor"],
        len(frequencies),
        num_mics=config["visualization"]["num_mics"],
    )
    visualized_room_filename = os.listdir(visualized_room_dir)[-1]
    visualized_room_filepath =  os.path.join(visualized_room_dir,visualized_room_filename)
    # Get measured sound field

    sf_sample = util.load_generated_soundfield(
        visualized_room_filepath, config["visualization"]["source"]
    )
    sf_gt = np.expand_dims(copy.deepcopy(sf_sample), axis=0)
    initial_sf = np.expand_dims(sf_sample, axis=0)

    # Get mask samples
    mask = mask_generator.sample()
    mask = np.expand_dims(mask, axis=0)

    # Preprocessing
    irregular_sf, mask = util.preprocessing(
        config["dataset"]["factor"], initial_sf, mask
    )

    # Save range to allow colorbar
    util.save_pressure_range(sf_gt, visualization_path)
    print("\tPressure range saved")
    # Scale ground truth sound field
    sf_gt = util.scale(sf_gt)

    print("\nPlotting Ground Truth Sound Field Scaled...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            sf_gt[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Ground_Truth.png"),
        )

    print("\nPlotting Irregular Sound Field...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            irregular_sf[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Irregular_SF.png"),
        )

    print("\nPlotting Mask...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            mask[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Mask.png"),
        )

    pred_sf = model.predict([irregular_sf, mask])

    print("\nPlotting Predicted Sound Field...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            pred_sf[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Pred_SF.png"),
        )
def visualize_multiple_mics(config_path):
    """Plot predictions of trained model on real data, using different
    number of mics.

        Args:
        config_path: string

    """

    config = util.load_config(config_path)
    searcher_path = "/content/gdrive/MyDrive/GitHub/sound-field-neural-network-58584a7774ecc997dec3663400843864cb6b79ed/config/searcher.json"
    searcher_config = util.load_config(searcher_path)

    print("Loaded configuration from: %s" % config_path)

    frequencies = util.get_frequencies()

    session_dir = config_path[: config_path.rfind("\\") + 1]

    checkpoint_path = get_general_checkpoint_path(
        session_dir=session_dir, number_checkpoint=-1
    )
    if not checkpoint_path:
        print("Error: No checkpoint found in same directory as configuration file.")
        return

    model = sfun.SFUN(config, train_bn=False)

    for num_mics in searcher_config["visualization"]["num_mics"]:
        visualization_path = os.path.join(session_dir, f"visualization_{num_mics}")
        if not os.path.exists(visualization_path):
            os.makedirs(visualization_path)

        filepath = os.path.join(
            config["dataset"]["path"], "real_soundfields", "RoomB_soundfield.mat"
        )

        mask_generator = data.MaskGenerator(
            config["dataset"]["xSamples"] // config["dataset"]["factor"],
            config["dataset"]["ySamples"] // config["dataset"]["factor"],
            len(frequencies),
            num_mics=num_mics,
        )

        # Get measured sound field
        sf_sample = util.load_RoomB_soundfield(
            filepath, config["visualization"]["source"]
        )
        sf_gt = np.expand_dims(copy.deepcopy(sf_sample), axis=0)
        initial_sf = np.expand_dims(sf_sample, axis=0)

        # Get mask samples
        mask = mask_generator.sample()
        mask = np.expand_dims(mask, axis=0)

        # preprocessing
        irregular_sf, mask = util.preprocessing(
            config["dataset"]["factor"], initial_sf, mask
        )

        # Save range to allow colorbar
        util.save_pressure_range(sf_gt, visualization_path)
        print("\tPressure range saved")
        # Scale ground truth sound field
        sf_gt = util.scale(sf_gt)

        print("\nPlotting Ground Truth Sound Field Scaled...")
        for num_freq, freq in enumerate(frequencies):
            print("\tat frequency " + str(freq))
            util.plot_2D(
                sf_gt[0, ..., num_freq],
                os.path.join(visualization_path, str(freq) + "_Hz_Ground_Truth.png"),
            )

        print("\nPlotting Irregular Sound Field...")
        for num_freq, freq in enumerate(frequencies):
            print("\tat frequency " + str(freq))
            util.plot_2D(
                irregular_sf[0, ..., num_freq],
                os.path.join(visualization_path, str(freq) + "_Hz_Irregular_SF.png"),
            )

        print("\nPlotting Mask...")
        for num_freq, freq in enumerate(frequencies):
            print("\tat frequency " + str(freq))
            util.plot_2D(
                mask[0, ..., num_freq],
                os.path.join(visualization_path, str(freq) + "_Hz_Mask.png"),
            )

        pred_sf = model.predict([irregular_sf, mask])

        print("\nPlotting Predicted Sound Field...")
        for num_freq, freq in enumerate(frequencies):
            print("\tat frequency " + str(freq))
            util.plot_2D(
                pred_sf[0, ..., num_freq],
                os.path.join(visualization_path, str(freq) + "_Hz_Pred_SF.png"),
            )


def predict_soundfield(config_path):
    """Plot predictions of trained model on real data.

    Args:
    config_path: string

    """

    config = util.load_config(config_path)
    print("Loaded configuration from: %s" % config_path)

    frequencies = util.get_frequencies()

    session_dir = config_path[: config_path.rfind("\\") + 1]

    checkpoint_path = get_general_checkpoint_path(
        session_dir=session_dir, number_checkpoint=-1
    )
    if not checkpoint_path:
        print("Error: No checkpoint found in same directory as configuration file.")
        return
    model = sfun.SFUN(config, train_bn=False)

    visualization_path = os.path.join(session_dir, "prediction", "visualization")
    if not os.path.exists(visualization_path):
        os.makedirs(visualization_path)

    filepath = config["prediction"]["predicted_file_path"]
    # Implementar tratamento de erro caso dê ""
    mask_generator = data.MaskGenerator(
        config["dataset"]["xSamples"] // config["dataset"]["factor"],
        config["dataset"]["ySamples"] // config["dataset"]["factor"],
        len(frequencies),
        num_mics=config["prediction"]["num_mics"],
    )

    # Get measured sound field
    sf_sample = util.load_generated_soundfield(
        filepath, config["visualization"]["source"]
    )

    sf_gt = np.expand_dims(copy.deepcopy(sf_sample), axis=0)
    initial_sf = np.expand_dims(sf_sample, axis=0)

    # Get mask samples
    mask = mask_generator.sample()
    mask = np.expand_dims(mask, axis=0)

    # preprocessing
    irregular_sf, mask = util.preprocessing(
        config["dataset"]["factor"], initial_sf, mask
    )

    # Save range to allow colorbar
    util.save_pressure_range(sf_gt, visualization_path)
    print("\tPressure range saved")
    # Scale ground truth sound field
    sf_gt = util.scale(sf_gt)

    print("\nPlotting Ground Truth Sound Field Scaled...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            sf_gt[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Ground_Truth.png"),
        )

    print("\nPlotting Irregular Sound Field...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            irregular_sf[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Irregular_SF.png"),
        )

    print("\nPlotting Mask...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            mask[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Mask.png"),
        )

    pred_sf = model.predict([irregular_sf, mask])

    print("\nPlotting Predicted Sound Field...")
    for num_freq, freq in enumerate(frequencies):
        print("\tat frequency " + str(freq))
        util.plot_2D(
            pred_sf[0, ..., num_freq],
            os.path.join(visualization_path, str(freq) + "_Hz_Pred_SF.png"),
        )
    
    util.save_predict_mat(original_file_path=filepath,
                          pred_sf=pred_sf,
                          save_path=visualization_path)
