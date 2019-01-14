import itertools
import numpy
import tensorflow

def double_lu(alpha=-2):
    return lambda x: tensorflow.where(tensorflow.greater(x, 0), x / alpha, x * alpha)

class InvertibleAutoencoder():
    def __init__(self, input_layer):
        self.outputs = [input_layer]
        
    def add_dense(self,
                  kernel,
                  filters,
                  strides,
                  activation):
        input_shape = numpy.prod(kernel)
        output_shape = numpy.prod(filters)
        w_shape = input_shape * (input_shape + 1) / 2
        with tensorflow.variable_scope('layer_{0}'.format(len(self.outputs))):
            W = tensorflow.get_variable('weights', (w_shape,),
                                initializer=tensorflow.constant_initializer(1.0))
            s = 2 * W / (W ** 2 + 1)
            c = (1 - W ** 2) / (1 + W ** 2)
            j = 1
            i = 0
            final_shape = max(input_shape, output_shape)
            final_rotation = tensorflow.eye(final_shape)
            for c0, s0 in zip(tensorflow.unstack(c), tensorflow.unstack(s)):
                if i == j or i >= output_shape:
                    j += 1
                    i = 0
                if j >= output_shape:
                    break
                givens_matrix = tensorflow.SparseTensor(indices=[[i, i], [j, j], [i, j], [j, i]],
                                values=tensorflow.stack([c0, c0, -s0, s0]),
                                                        dense_shape=[final_shape, final_shape])
                final_rotation = tensorflow.sparse_tensor_dense_matmul(givens_matrix,
                                                                       final_rotation)
                i += 1
            final_rotation = tensorflow.slice(final_rotation, [0,] * 2, [output_shape, input_shape])
            bias = tensorflow.get_variable('bias_{0}'.format(len(self.outputs)),
                                output_shape,
                                initializer=tensorflow.constant_initializer(0.0))
            
            # convolution
            padded = tensorflow.pad(self.outputs[-1], [[k // 2, k // 2] for k in kernel])
            output_shape = [dim // stride for dim, stride in zip(self.outputs[-1].shape, strides)] + filters
            output = tensorflow.Variable(tensorflow.zeros(shape=output_shape, dtype=tensorflow.float32))
            for i in itertools.product(*[range(0, dim, stride) for dim, stride in zip(self.outputs[-1].shape, strides)]):
                k = [i0 // s for i0, s in zip(i, strides)]
                f = tensorflow.matmul(final_rotation,
                                      tensorflow.reshape(tensorflow.slice(padded,
                                                                          i,
                                                                          kernel),
                                                         [input_shape, 1]))
                tensorflow.scatter_nd_update(output, k, tensorflow.squeeze(f))
            self.outputs.append(activation(output + bias))

if __name__ == '__main__':
    a = InvertibleAutoencoder(tensorflow.placeholder(tensorflow.float32, shape=(128, 128)))
    a.add_dense([2, 2], [4], [2, 2], double_lu(2))
