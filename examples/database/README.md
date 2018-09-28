# Dispatch Tasks

1. This script (`dispatch_tasks.py`) creates 3 tasks (`task_script.py`) with different hyper-parameters and a dataset and pushes these tasks into the database. 
2. On your GPU servers (for testing, it can be a new terminal on your local machine), run tasks as shown in `run_tasks.py`. 
This script pulls and runs pending tasks, and saves the models and results to the database.
3. When all tasks complete, the dispatcher (`dispatch_tasks.py`)  then selects the best model according to its accuracy.


# Save and load models

- `task_script.py` shows how to save model.
- `dispatch_tasks.py ` shows how to find and load the model with the best testing accuracy.

# Save and load datasets

- `dispatch_tasks.py ` shows how to save a dataset.
- `task_script.py ` show how to find and load a dataset.

#### More information in the online documentation.