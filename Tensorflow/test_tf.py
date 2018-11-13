import tensorflow as tf
import numpy as np

if __name__=='__main__':
    sess = tf.Session()
    
    Data = tf.placeholder(dtype = tf.float32, shape = (None, 784))

    Omega = tf.Variable(tf.random_normal((784,4),stddev = 0.25))
    
    Result = tf.matmul(Data,Omega)

    Target = tf.placeholder(dtype = tf.float32, shape = (None, 4))

    loss = tf.reduce_mean((Result-Target)**2)

    data_val = np.ones((100,784))
    target_val = np.ones((100,4))

    optimizer = tf.train.AdamOptimizer(1e-2)

    global_step=None
    var_list=None
    gate_gradients=1
    aggregation_method=None
    colocate_gradients_with_ops=False
    name=None
    grad_loss=None

    grads_and_vars = optimizer.compute_gradients(
        loss, var_list=var_list, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        grad_loss=grad_loss)

    vars_with_grad = [v for g, v in grads_and_vars if g is not None]
    if not vars_with_grad:
      raise ValueError(
          "No gradients provided for any variable, check your graph for ops"
          " that do not support gradients, between variables %s and loss %s." %
          ([str(v) for _, v in grads_and_vars], loss))


    train_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step, name=name)

    initializer = tf.global_variables_initializer()

    sess.run(initializer)

    for _ in range(100):
        
        result, _ = sess.run([loss, train_step], feed_dict= {Data: data_val, Target: target_val})

        print(result)



