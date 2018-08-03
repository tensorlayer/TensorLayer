"""
Runs this script on local machine, it will push dataset and tasks into database.
"""
import time
import tensorlayer as tl
import tensorflow as tf

tl.logging.set_verbosity(tl.logging.DEBUG)
# tf.logging.set_verbosity(tf.logging.DEBUG)

## connect to database
db = tl.db.TensorHub(ip='localhost', port=27017, dbname='temp', project_key='tutorial')

## delete existing tasks, models and datasets in this project
db.del_task()
db.del_model()
db.del_dataset()

## save dataset into database, then allow  other servers to use it
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
db.save_dataset((X_train, y_train, X_val, y_val, X_test, y_test), 'mnist', description='handwriting digit')

## push tasks into database, then allow other servers pull tasks to run
db.push_task(
    task_key='mnist', script='task_script.py', hyper_parameters=dict(n_units1=800, n_units2=800),
    result_key=['test_accuracy'], description='800-800'
)

db.push_task(
    task_key='mnist', script='task_script.py', hyper_parameters=dict(n_units1=600, n_units2=600),
    result_key=['test_accuracy'], description='600-600'
)

db.push_task(
    task_key='mnist', script='task_script.py', hyper_parameters=dict(n_units1=400, n_units2=400),
    result_key=['test_accuracy'], description='400-400'
)

## wait for tasks to finish
while db.check_unfinished_task(task_key='mnist'):
    print("waiting runners to finish the tasks")
    time.sleep(1)

## get the best model
print("all tasks finished")
sess = tf.InteractiveSession()
net = db.find_one_model(sess=sess, model_key='mlp', sort=[("test_accuracy", -1)])
print("the best accuracy {} is from model {}".format(net._test_accuracy, net._name))
