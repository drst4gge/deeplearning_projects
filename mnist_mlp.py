import time

print("importing TensorFlow bro")
import tensorflow as tf
from silence_tensorflow import silence_tensorflow
silence_tensorflow()

N_batch = 200

print("loading mnist dataset bro")
import tensorflow_datasets as tfds
(ds_train, ds_test), ds_info = tfds.load(
	'mnist',
	split=['train', 'test'],
	shuffle_files=True,
	as_supervised=True,
	with_info=True,
)

fig=tfds.show_examples(ds_test,tfds.builder('mnist').info)

print("pre-processing the training data bro")
def normalize_img(image, label):
	return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
	normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(N_batch)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

print("now we pre-process the verification data bro")
ds_test = ds_test.map(
	normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(N_batch)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

print("now we make the model bro")
model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(64, activation='relu'),
	tf.keras.layers.Dense(32, activation='relu'),
	tf.keras.layers.Dense(10)
])

print("compiling dis model now bro")
model.compile(
	optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

model.summary(
	line_length=None,
	positions=None,
	print_fn=None,
	expand_nested=False,
	show_trainable=False,
)

print("training bro, lifting weights on dis model bro with protein")
t1=time.perf_counter()
model.fit(
	ds_train,
	epochs=24,
	validation_data=ds_test,
	validation_freq=6,
	verbose=1,
)
trainingtime=time.perf_counter()-t1
print("it took this much time to train bro",trainingtime)



























