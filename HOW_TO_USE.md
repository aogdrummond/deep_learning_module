## Usage


There are six operating modes: training, simulated data evaluation, real data evaluation, visualization, evaluation and prediction. A JSON-formatted configuration file defines the parameters of an existing or a new session. The structure of this configuration file is described [here](config/config.md).

* Open Anaconda Prompt
* Move to this directory with command `"cd <path_to_deep_learning_module>"`
* Activate virtual environment with `"conda activate deep_learning_module"`
* Type the command for desired operating mode

## Operating modes:

#### **Training**

* Command: `python main.py --mode train --config <path_to_initial_config_file>`. Training will begin after the dataset is prepared for training.

Training sessions are differentiated from one another by a session ID defined in `initial_config.json`. All artifacts generated during training are placed in a folder corresponding to this session ID in the `neural_network_sessions/` folder, inside the storage directory.


Alternatively, there is a pre-trained model available in the `sessions/` folder (called session_0) for quickly trying out with test data.

#### **Simulated Data Evaluation**

* Command: `python main.py --mode sim-eval --config <path_to_sessions_config_file>`. Note: --config should contain the path to a configuration file *in a session folder, on default saved in the storage*.

To evaluate a model we present every room in the simulated test set with several microphones locations to the most recent checkpoint present in a session folder, and calculate the Normalized Mean Square Error (NMSE) and the Structural Similarity (SSIM) over all analyzed frequencies.

A new directory named `simulated_data_evaluation` is created inside the session folder. It contains a `.csv` file (for each room) containg each individual result and plots showing the performance of the model regarding the metrics and the number of microphones. 

#### **Real Data Evaluation**

* Command: `python main.py --mode real-eval --config <path_to_config_file>`. Note: --config should contain the path to a configuration file *in a session folder*.

To evaluate a model we present the measured real room with several microphones locations to the most recent checkpoint present in a session folder, and calculate the Normalized Mean Square Error (NMSE) and the Structural Similarity (SSIM) over all analyzed frequencies.

A new directory named `real_data_evaluation` is created inside the session folder. It contains a `.csv` file containg each individual result and plots for both source locations showing the performance of the model regarding the metrics and the number of microphones.

#### **Evaluation**

* Command: `python main.py --mode general-evaluation --config <path_to_config_file>`


Runs the model evaluation through NMSE and SSIM metrics, evaluating the average performance from the `.csv` files created at `simulated_data_evaluation`. It will create in the same directory containing the csv's a folder called "average_performance", where are saved the plots of NMSE and SSIM averaged. Lastly, a `.jpg` file containing the plot with history of training and validation loss during the training will be saved.

Requires `sim-eval` to be run before.

#### **Visualization**

* Command: `python main.py --mode visualize-real-room --config <path_to_config_file>`. Note: --config should contain the path to a configuration file *in a session folder*.

We may wish to visualize the sound field reconstruction on real data.

A new directory named `visualization` is created inside the session folder. It contains images of the ground truth sound field, the irregular sound field gathered, the mask, and the predicted sound field for each analyzed frequency. It is used the most recent checkpoint present in the session folder.

#### **Prediction**

* Command: `python main.py --mode predict-sim --config <path_to_config_file>`. Note: --config should contain the path to a configuration file *in a session folder*.

We may wish to visualize the sound field reconstruction on simulated data generated on MATLAB or Femder.

You just need to write on session's "config.json" file the path to the `.mat` file containing the original soundfield. It will run the prediction and save images with predicted soundfields, as well as a `.mat` with the prediction, in the same folder, so it is recommended to use an empty folder inside the session.

#### **Compare prediction for different number of microphones**

* Command: `python main.py --mode visualize-many-mics --config <path_to_config_file>`. Note: --config should contain the path to a configuration file *in a session folder*.

To plot the prediction using different number of microphones, but the same model, you just need to create a folder "visualization", if it is not created, and save in it the `.mat` file with the information about the simulated soundfield to be predicted. The visualization with a number of mics will be saved in the same folder. 

#### **Compare soundfields from prediction with ground truth**

* Command: `python main.py --mode compare-soundfields --config <path_to_config_file>`. Note: --config should contain the path to a configuration file *in a session folder*.

Requires `visualize-real-room` to be run before.