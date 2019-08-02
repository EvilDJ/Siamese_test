import numpy as np
#np.ceil 取大于此值的最小值，np.floor 取小于此值的最大值
def spp(data, bins):
    shape = data.get_shape().as_list()
    kize_out = []
    stride_out = []
    for i in range(bins):
        i = i + 1
        kize = [1, np.ceil(shape[1] / i + 1).astype(np.int32), np.ceil(shape[1] / i + 1).astype(np.int32), 1]
        stride = [1, np.floor(shape[1] / i + 1).astype(np.int32), np.floor(shape[2] / i + 1).astype(np.int32), 1]
        # pool_out.append(tf.nn.max_pool(data, ksize=kize, strides=stride, padding='SAME'))
        kize_out.append(kize)
        stride_out.append(stride)
    return kize_out,stride_out
