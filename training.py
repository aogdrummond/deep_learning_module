# Sound field reconstruction in rooms: inpainting meets superresolution - 17.12.2019
# Training.py

from __future__ import division
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

    model = sfun.SFUN(config,train_bn=True)
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

