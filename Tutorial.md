# Environment setting

## Explanation bla bla bla

The module is intended to be used from Anaconda Prompt

# Training

For the first training to be started it is required to set on config\config.json only the parameters storage.path and dataset.name to indicate which dataset will be used to train, using all the parameters as default.

For a customized training it can still be changed batch_size, learning rate("lr"), number of epochs ran and so on...

Certify yourself that the parameter "session_id" is correctly set to a new id, since if it is the id of an existing session the training will continue the session from the its last checkpoint.


# Evaluation

