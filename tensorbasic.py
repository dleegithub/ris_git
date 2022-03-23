print("0323_b")
import tensorflow as tf
if not tf.test.is_built_with_cuda():
    print("not built with cuda")
import numpy as np
np.savetxt("testgpu.txt", tf.config.list_physical_devices('GPU'))


t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
result =  tf.concat([t1, t2], 0)
np.savetxt("test.csv", result, fmt='%.2f', delimiter=",")
