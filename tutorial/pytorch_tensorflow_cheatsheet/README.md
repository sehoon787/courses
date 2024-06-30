## Tensor Initialization
|                           |   PyTorch           |     Tensorflow         |        NumPy       |
|---------------------------|---------------------|------------------------|--------------------|
|                           |   torch.tensor()    |     tf.constant()      |   np.array()       |
|                           |   torch.full()      |     tf.fill()          |   np.full()        |
|                           |  torch.from_numpy() | tf.convert_to_tensor() |         -          |
|                           |  torch.range()      | tf.range()             |  np.arange()       |
|                           |  torch.zeors()      | tf.zeros()             |  np.zeros()        |
|                           |  torch.zeors_like() | tf.zeros_like()        |                    |
|                           |  torch.ones()       | tf.ones()              |  np.ones()         |
|                           |  torch.ones_like()  | tf.ones_like()         |                    |
|                           |  torch.nn.functional.one_hot()  | tf.one_hot()           |                    |


```
p0 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  dtype=torch.float64)
t1 = torch.tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
t0 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.dtypes.float32)
t1 = tf.constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
```
## Attributes
|                           |   PyTorch            |     Tensorflow      |        NumPy          |
|---------------------------|----------------------|---------------------|-----------------------|
|   Number of Dimensions    |   torch.Tensor.dim() |     tf.rank()       |   numpy.ndarray.ndim  |
|   Shape of a Tensor       |   torch.Tensor.size() or<br> torch.Tensor.shape |     tf.shape() or<br> tf.Tensor.shape    |   numpy.ndarray.shape  |

                                                              
## Tensor Arithmetic
|                           |   PyTorch                    |     Tensorflow       |        NumPy       |
|------------------------------|------------------------------|------------------------------|--------------------|
|                              |   torch.min()                |    tf.reduce_min()           |                    | 
|                              |   torch.max()                |    tf.reduce_max()   |                    | 
|                              |   torch.abs()                |    tf.abs()              |                    |
|                              |   torch.multiply()           |    tf.math.multiply()       |                    |
|                              |   torch.log()                |    tf.math.log()       |                    |
|                              |   torch.softmax()            |    tf.nn.softmax()   |                    |
|                              |   torch.log_softmax()        |    tf.nn.log_softmax()        |                    |
|                              |   torch.logaddexp()          | tfp.math.log_add_exp |                    |
|                              |   torch.where()              |    tf.where()        |                    |
|                              |   torch.greater_equal()      |    tf.math.greater_equal()    |                    |

# Tensor Manipulation
|                            |   PyTorch                   |     Tensorflow      |        NumPy        |
|----------------------------|-----------------------------|---------------------|---------------------|
| Type Conversion            |   torch.Tensor.type()       |     tf.cast()       | np.ndarray.astype() |
|                            |   torch.reshape() or<br>   torch.Tensor.view()      |     tf.reshape()    |                     |
|                            |   torch.cat()               |     tf.concat()     |                     |
|                            |   torch.stack()             |     tf.stack()      |                     |
|                            |   torch.unbind()            |     tf.unstack()    |                     |
|                            |   torch.repeat() or<br> torch.tile()    |     tf.tile()       |                     |
|                            |   torch.unsqueeze() or<br> torch.expand()        |     tf.expand_dims()|  np.expand_dims()   |


## Indexing
* tf.gather
* tf.scatter_nd_update



## Random Number

|                            |   PyTorch                   |     Tensorflow        |        NumPy         |
|----------------------------|-----------------------------|-----------------------|----------------------|
|  Random Seed               | torch.manual_seed()         | tf.random.set_seed()  |  np.random.seed()    |
|  Gaussian Random Number    | torch.randn()     | tf.random.normal()    |  np.random.normal()    |
|  Random Integer            | torch.randint()   | tf.random.randint()   |  np.random.randint()   |




# References


* PyTorch Cheat Sheet https://pytorch.org/tutorials/beginner/ptcheat.html
* PyTorch Tensorflow CheatSheet  https://github.com/cdeboeser/tensorflow-torch-cheatsheet
* https://bladejun.tistory.com/145
