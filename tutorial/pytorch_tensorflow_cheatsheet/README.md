

# Tensor Initialization
|                           |   PyTorch           |     Tensorflow         |        NumPy       | 
|---------------------------|---------------------|------------------------|--------------------|
|                           |   torch.tensor()    |     tf.constant()      |   np.array()       |
|                           |   torch.full()      |     tf.fill()          |   np.full()        |
|                           |  torch.from_numpy() | tf.convert_to_tensor() |         -          |   


```
p0 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  dtype=torch.float64)
t1 = torch.tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
t0 = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.dtypes.float32)
t1 = tf.constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
```
# Attributes
|                           |   PyTorch            |     Tensorflow      |        NumPy          | 
|---------------------------|----------------------|---------------------|-----------------------|
|   Number of Dimensions    |   torch.Tensor.dim() |     tf.rank()       |   numpy.ndarray.ndim  |



# Tensor Manipulation
|                           |   PyTorch         |     Tensorflow      |        NumPy       | 
|---------------------------|-------------------|---------------------|--------------------|
|                           |   torch.reshape() |     tf.reshape()    |                    |
|                           |   torch.cat()     |     tf.concat()     |                    |
|                           |   torch.stack()   |     tf.stack()      |                    |

# Indexing
* tf.gather
* tf.scatter_nd_update




# Random Number

|                           |   PyTorch       |     Tensorflow      |        NumPy         |
|---------------------------|-----------------|---------------------|----------------------|
|  Gaussian Random Number   | torch.randn()     | tf.random.normal()    |  np.random.normal()    |
|  Random Integer           | torch.randint()   | tf.random.randint()   |  np.random.randint()   | 




# References


PyTorch Cheat Sheet https://pytorch.org/tutorials/beginner/ptcheat.html
PyTorch Tensorflow CheatSheet  https://github.com/cdeboeser/tensorflow-torch-cheatsheet
