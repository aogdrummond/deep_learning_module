# Configuring a session
----

### **"initial_config.json" must be used only to create a new session. It will create session's folder with its own config inside the storage module, therefore to any other than training use session's "config.json"**


### **Storage**
* **path**: *string* Path to the storage integrated for the neural network and room simulation modules, aiming to store the data outputed from the modules

### **Dataset**
Where to find and how to handle the data.
* **name**: *string* Path to dataset
* **num_freq**: *int" Number of frequencies from sample's soundfield used in the neural network
* **factor**: *int* Value of the factor applied during downsampling and upsampling
* **xSamples**: *int* Number of ground truth samples along the x-axis in the room. (32 if using the provided dataset)
* **ySamples**: *int* Number of ground truth samples along the y-axis in the room. (32 if using the provided dataset)

### **Evaluation**
How evaluation will be performed.
* **min_mics**: *int* Minimum number of microphones placed in a room to evaluate the model.
* **max_mics**: *int* Maximum+1 number of microphones placed in a room to evaluate the model.
* **step_mics**: *int* Spacing between the value of the number of microphones placed.
* **num_comb**: *int* Number of different irregular patterns tested with a fix amount of microphones.
* **frequencies**: *list* List of frequencies of interest when the soundfields are plotted
### **Training**
How training will be performed.
* **batch_size**: *int* Number of samples in a batch
* **num_epochs**: *int* Number of epochs to train for
* **num_steps_val**: *int* Total number of steps (batches of samples) to yield from validation generator before stopping at the end of every epoch.
* **num_steps_train**: *int* Total number of steps (batches of samples) to yield from validation generator before stopping at the end of every epoch.
* **session_id**: *int* Numerical identifier for this session
* **lr**: *float* Learning rate
* **loss**: *float* 
  * **valid_weight**: *float* Weight given to the loss term considering microphone position predictions.
  * **hole_weight**: *float* Weight given to the loss term considering non-microphone position predictions.
 

### **Visualization**
Visualization of how it predicts a real room soundfield. In this case the ground truth is the result of measurement of a real room, aiming to check the performance on real cases
* **num_mics**: *int* Number of microphones randomly located in the real room.
* **source**: *int* Numerical identifier of the source location. Must be either 0 or 1.

### **Prediction**

Prediction of soundfields from simulated files.
 * **predicted_file_path**: *str* Complete path for the file which will be predicted and compared through trained model 
 * **num_mics**: *int* Number of microphones randomly located in the predicted room.

 ### **Note:** Python uses forward slash ("/") as delimiter on pathes, while Windows by defaults uses backslash ("\\"). It may need to be adapted on the path, mainly on "Prediction", since it receives the path directly. Change it if a JSON Decode Error is raised.