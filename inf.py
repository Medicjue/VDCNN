import numpy as np
import tensorflow as tf
from data_helper import data_helper





# Data Preparation
data_helper = data_helper(sequence_max_length=1024)
tf.reset_default_graph()
#sess = tf.Session()
sess = tf.InteractiveSession()

saver = tf.train.import_meta_graph('mdl/vdcnn_small.ckpt.meta')

sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

saver.restore(sess, "mdl/vdcnn_small.ckpt")
all_vars = tf.trainable_variables()
for v in all_vars:
    print("%s with value %s" % (v.name, sess.run(v)))
    
print('Restore Session completed!')

graph = tf.get_default_graph()
#input_x = tf.placeholder(tf.int32, [None, 1024], name="input_x")
#is_training = tf.placeholder(tf.bool, name="is_training")
is_training = graph.get_tensor_by_name("is_training:0")

#print(graph.get_operations())
#print(graph.get_all_collection_keys())
input_x = graph.get_tensor_by_name("input_x:0")
input_y = graph.get_tensor_by_name("input_y:0")
#predictions = graph.get_tensor_by_name("loss/predictions")
#input_x = graph.get_operation_by_name("input_x")
#predictions = graph.get_operation_by_name("loss/predictions")

predictions = graph.get_tensor_by_name("loss/predictions:0")

probabilities = graph.get_tensor_by_name("fc3/fc3_tmp:0")


#predictions.eval(session=sess)

inf_data = data_helper.load_csv_file('data_collect/vdcnn_tf_inf.csv', num_classes=2)

feed_dict = {input_x: inf_data[0], 
             is_training: False}
preds = sess.run([predictions], feed_dict)


print(preds)
