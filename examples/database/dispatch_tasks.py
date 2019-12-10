"""
A sample script that shows how to distribute multiple tasks to multiple machine
using the database module.

"""
import time

import tensorflow as tf

import tensorlayer as tl

tl.logging.set_verbosity(tl.logging.DEBUG)
# tf.logging.set_verbosity(tf.logging.DEBUG)

# connect to database
db = tl.db.TensorHub(ip='localhost', port=27017, dbname='temp', project_name='tutorial')

# delete existing tasks, models and datasets in this project
db.delete_tasks()
db.delete_model()
db.delete_datasets()

# save dataset into database, then allow  other servers to use it
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
db.save_dataset((X_train, y_train, X_val, y_val, X_test, y_test), 'mnist', description='handwriting digit')

# push tasks into database, then allow other servers pull tasks to run
db.create_task(
    task_name='mnist', script='task_script.py', hyper_parameters=dict(n_units1=800, n_units2=800),
    saved_result_keys=['test_accuracy'], description='800-800'
)

db.create_task(
    task_name='mnist', script='task_script.py', hyper_parameters=dict(n_units1=600, n_units2=600),
    saved_result_keys=['test_accuracy'], description='600-600'
)

db.create_task(
    task_name='mnist', script='task_script.py', hyper_parameters=dict(n_units1=400, n_units2=400),
    saved_result_keys=['test_accuracy'], description='400-400'
)

# wait for tasks to finish
while db.check_unfinished_task(task_name='mnist'):
    print("waiting runners to finish the tasks")
    time.sleep(1)

# get the best model
print("all tasks finished")
net = db.find_top_model(model_name='mlp', sort=[("test_accuracy", -1)])
print("the best accuracy {} is from model {}".format(net._test_accuracy, net._name))
