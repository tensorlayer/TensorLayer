"""
Run this script on servers, it will monitor the database and run tasks when
someone push a task to the database.
"""

import time
import tensorlayer as tl

# tl.logging.set_verbosity(tl.logging.DEBUG)

## connect to database
db = tl.db.TensorHub(
    ip='localhost', port=27017, dbname='temp', username=None, password='password', project_key='tutorial'
)

## monitor the database and pull tasks to run
while True:
    print("waiting task from distributor")
    db.run_one_task(task_key='mnist', sort=[("time", -1)])
    time.sleep(1)
