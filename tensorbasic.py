
import tensorflow as tf
print(tf.test.is_gpu_available)
import numpy as np
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
result =  tf.concat([t1, t2], 0)
np.savetxt("test.csv", result, fmt='%.2f', delimiter=",")
