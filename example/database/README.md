# Task Distribution

1. Run task distributor (`task_distributor.py`) on your local machine to 3 tasks (`task_script.py`) with different hyper-parameters and a dataset to the database. 
2. On your GPU servers (for testing, it can be a new terminal on your local machine), run task runners (`task_runner.py`). 
This will pull and run the unfinished tasks, and save the models and results to the database.
3. When all tasks are finished, the distributor (`task_distributor.py`) will automatically get the best model according to its accuracy.


# Save and load model

- `task_script.py` shows how to save model.
- `task_distributor.py` show how to find and load the model with the best testing accuracy.

# Save and load dataset

- `task_distributor.py ` shows how to save a dataset.
- `task_script.py ` show how to find and load a dataset.

#### More information in the online documentation.