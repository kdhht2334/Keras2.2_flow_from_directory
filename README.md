## Keras2.2_flow_from_directory
Method to extend `flow_from_directory` function of Keras library


In this repository, we show how to extend the functionality of `flow_from_directory` function of Keras.

![Keras](./images/keras.jpg)

-----
### Usage

+1. Add your training options to `allowd_class_mode` of `DirectoryIterator` class in `directory_iterator.py`.

(Full path is `anaconda3/envs/tensorflow/lib/python3.6/site-packages/keras_preprocessing/image/directory_iterator.py`)

<br>

Like this.

```python
class DirectoryIterator(BatchFromFilesMixin, Iterator):
	allowed_class_modes = {'categorical', 'binary', 'sparse', 'input', 'input', 'colorize', 'kl_divergence_ILSVRC', 'kl_divergence_96', None}
    def __init__(self,
                 directory,
                 image_data_generator,
                 target_size=(256, 256),
                 dimension_ILSVRC=512,
                 dimension_96=96,
                     ...
```

<br>

+2. Add your method to `BatchFromFilesMixin` class in `iterator.py`.

(Full path is `anaconda3/envs/tensorflow/lib/python3.6/site-packages/keras_preprocessing/image/iterator.py`)

More specifically, you have to change `_get_batches_of_transformed_samples` functions to add some functionality.

<br>

Like this.

```python
def _get_batches_of_transformed_samples(self, index_array):
       
             ...
             
             
    # build batch of labels
    if self.class_mode == 'input':
        batch_y = batch_x.copy()
    elif self.class_mode in {'binary', 'sparse'}:
        batch_y = np.empty(len(batch_x), dtype=self.dtype)
        for i, n_observation in enumerate(index_array):
            batch_y[i] = self.classes[n_observation]
    elif self.class_mode == 'categorical':
        batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
        for i, n_observation in enumerate(index_array):
              batch_y[i, self.classes[n_observation]] = 1.
    elif self.class_mode == 'other':
         batch_y = self.data[index_array]
         
    #TODO(): Add new functionality :)
    elif self.class_mode == 'colorize':
        batch_x, batch_y = image_a_b_gen(batch_x)
    elif self.class_mode == 'kl_divergence_96':
        batch_y = np.random.normal(size=(len(batch_x), self.dimension_96))
    elif self.class_mode == 'kl_divergence_ILSVRC':
        batch_y = np.random.normal(size=(len(batch_x), self.dimension_ILSVRC))
    else:
        return batch_x
    return batch_x, batch_y
```

+3. Use `flow_from_directory` function in your source like this.

```python
# data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess)

train_generator2 = train_datagen.flow_from_directory(
        '%s/train/' % ds,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="colorize")

```
<br>
-----
### Milestone

[v] Add classification task example

[v] Add colorization task example

[v] Add Kullback-Leibler divergence example