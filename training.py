# Sound field reconstruction in rooms: inpainting meets superresolution - 17.12.2019
# Training.py

from __future__ import division

# import sys
# sys.path.append('util')
import os
import util.util as util
import data
import sfun


def create_new_session(config):
    """Creates a new folder to save all session artifacts to.
    Args:
    config: dict, session configuration parameters
    """
    sessions_storage_path = os.path.join(config["storage"]["path"],
                                         "neural_network_sessions")

    if "session_dir" not in config["training"]:

        current_session = "".join(["session_",str(config["training"]["session_id"])])
        session_dir_path = os.path.join(sessions_storage_path,current_session)
        config["training"]["session_dir"] = session_dir_path
        
        if not os.path.exists(session_dir_path):
            os.mkdir(session_dir_path)
        util.save_config(config["training"]["session_dir"], config)

    session_path = os.path.join(config["storage"]["path"], config["training"]["session_dir"])

    if not os.path.exists(session_path):
        os.mkdir(session_path)


def train(config_path):
    """Trains a model

    Args:
    config_path: string, path to a config.json file
    """

    # Load configuration
    if not os.path.exists(config_path):
        print("Error: No configuration file present at specified path.")
        return

    config = util.load_config(config_path)
    print("Loaded configuration from: %s" % config_path)

    if "session_dir" not in config["training"]:
        create_new_session(config)
    storage_session_path = os.path.join(config["storage"]["path"], 
                                        config["training"]["session_dir"])
    if not os.path.exists(storage_session_path):
        create_new_session(config)

    model = sfun.SFUN(config)
    dataset = data.Dataset(config).load_dataset()

    train_set_generator = dataset.get_random_batch_generator("train")
    val_set_generator = dataset.get_random_batch_generator("val")

    model.fit_model(
        train_set_generator,
        config["training"]["num_steps_train"],
        val_set_generator,
        config["training"]["num_steps_val"],
        config["training"]["num_epochs"],
    )


def model_finetuning(config_path: str):
    """Trains multiple sessions, always choosing the feature with
    best performance among those evaluated before variating the next
    feature

    Args:
    config_path: string, path to a config.json file
    """
    searcher_path = "/content/gdrive/MyDrive/GitHub/sound-field-neural-network-58584a7774ecc997dec3663400843864cb6b79ed/config/searcher.json"
    # Load configuration
    if not os.path.exists(config_path):
        print("Error: No configuration file present at specified path.")
        return

    config = util.load_config(config_path)
    searcher_config = util.load_config(searcher_path)
    print("Loaded configuration from: %s" % config_path)

    dataset = data.Dataset(config).load_dataset()
    train_set_generator = dataset.get_random_batch_generator("train")
    val_set_generator = dataset.get_random_batch_generator("val")

    i = 0
    id_sessions_trained = []
    evaluated_field = "batch_size"
    for hp, epochs in zip(
        searcher_config["training"][evaluated_field],
        searcher_config["training"]["num_epochs"],
    ):

        config["training"][evaluated_field] = hp
        config["training"]["num_epochs"] = epochs
        config["training"]["session_id"] = searcher_config["training"]["session_id"] + i
        id_sessions_trained.append(config["training"]["session_id"])
        print("Configuration overwritten")
        # Create session directory
        if "session_dir" not in config["training"] or os.path.exists(
            config["training"]["session_dir"]
        ):
            create_new_session(config)
        # Skips existing sessions directly to next stage
        else:
            continue

        model = sfun.SFUN(config)
        model.fit_model(
            train_set_generator,
            config["training"]["num_steps_train"],
            val_set_generator,
            config["training"]["num_steps_val"],
            config["training"]["num_epochs"],
        )
        i += 1

    root_path = "/content/gdrive/MyDrive/GitHub/sound-field-neural-network-58584a7774ecc997dec3663400843864cb6b79ed/sessions/"
    model_path = best_model_path(id_sessions_trained, root_path)
    config["training"]["session_id"] = str(model_path.split("/")[-1])

    id_sessions_trained = []
    config = util.load_config("".join([model_path, "/config.json"]))

    # STEP 2
    for hp1, hp2, epochs in zip(
        searcher_config["training"]["num_steps_train"],
        searcher_config["training"]["num_steps_val"],
        searcher_config["training"]["num_epochs"],
    ):

        config["training"]["num_steps_train"] = hp1
        config["training"]["num_steps_val"] = hp2
        config["training"]["num_epochs"] = epochs
        config["training"]["session_id"] = searcher_config["training"]["session_id"] + i
        id_sessions_trained.append(config["training"]["session_id"])
        print("Configuration overwritten")
        # Create session directory
        if "session_dir" not in config["training"] or os.path.exists(
            config["training"]["session_dir"]
        ):
            create_new_session(config)

        model = sfun.SFUN(config)

        model.fit_model(
            train_set_generator,
            config["training"]["num_steps_train"],
            val_set_generator,
            config["training"]["num_steps_val"],
            config["training"]["num_epochs"],
        )
        i += 1


def best_model_path(id_sessions_trained: list, root_path: str) -> str:
    """
    Choose the session with best performance among those with id contained in the
    list "id_sessions_trained", according to its val_loss only
    """

    sessions_pathes = [
        "".join([root_path, "session_", str(session_id)])
        for session_id, root_path in zip(
            id_sessions_trained, [root_path] * len(id_sessions_trained)
        )
    ]
    best_loss, best_path = get_session_loss(sessions_pathes[0])

    for session_path in sessions_pathes[1:]:
        (val_loss, session_path) = get_session_loss(sessions_pathes[3])
        if val_loss < best_loss:
            best_path = session_path

    return best_path


def get_session_loss(session_path: str) -> tuple:
    """
    Function to obtain the validation loss and session path from the last checkpoint
    saved in the directory contained on "session_path"
    """

    best_checkpoint_name = os.listdir("".join([session_path, "/checkpoints"]))[-1]
    val_loss = float(best_checkpoint_name[17:21])

    return (val_loss, session_path)
