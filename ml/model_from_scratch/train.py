from __future__ import annotations
from pathlib import Path
from time import strftime
from subprocess import run
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
import tensorflow.keras.utils as utils
import tensorflow.keras.initializers as initializers
import tensorflow.keras.callbacks as callbacks
import tensorflow.image as image
import tensorflow.data as data

def save_name() -> str:
    """Generate a save name from the current date."""
    return strftime("%Y_%m_%d_%H_%M")

def save_stuff(model: Model, savename: str = save_name()) -> None:
    """Save the model, the history and the category names."""
    model.model.save(f"saves/{savename}")

    images = Path("categorized_images")
    folders = images.iterdir()
    categories = []
    for folder in folders:
        if folder.is_dir():
            categories.append(str(folder).split('/')[1])
    categories.sort()

    run(['mv', 'history.csv', f"saves/{savename}"])
    run(['mv', 'description.txt', f"saves/{savename}"])

    with open(f"saves/{savename}/categories.txt", 'w') as fout:
        print(categories, file=fout)

train, valid = utils.image_dataset_from_directory(
        'categorized_images',
        batch_size=32,
        label_mode='categorical',
        image_size=(395, 395),
        validation_split=0.25,
        subset='both',
        seed=3759,
)

train = train.cache().prefetch(buffer_size = data.AUTOTUNE)
valid = valid.cache().prefetch(buffer_size = data.AUTOTUNE)

hflip = train.map(lambda x, y: (image.flip_left_right(x), y))
brightness = train.map(lambda x, y: (image.random_brightness(x, max_delta=0.25), y))
hue = train.map(lambda x, y: (image.random_hue(x, max_delta=0.25), y))
train = train.concatenate(hflip)
train = train.concatenate(brightness)
train = train.concatenate(hue)

class Model:
    def __init__(self, input_size):
        self.description = ""
        self.model = tf.keras.Sequential()
        # depth, frame size are first 2 args
        # First layer of a Sequential Model should get input_shape as arg
        # Input: 395 x 395 x 3
        self.model.add(layers.Conv2D(
                12, 
                17, 
                strides=7, 
                activation=activations.relu,
                input_shape=input_size,
        ))
        # Size: 55 x 55 x 12
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D(
                pool_size=3,
                strides=2,
        ))
        # Size: 27 x 27 x 12
        self.model.add(layers.Conv2D(
                18,
                3,
                strides=1,
                activation=activations.relu,
        ))
        # Size: 25 x 25 x 18
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool2D(
                pool_size=3,
                strides=2,
        ))
        # Size: 12 x 12 x 18
        self.model.add(layers.Flatten())
        # Size: 2592
        self.model.add(layers.Dense(
                512, 
                activation=activations.relu,
                ))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(
                128, 
                activation=activations.relu,
                ))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(
                32, 
                activation=activations.relu,
                ))
        # Size of last Dense layer MUST match # of classes
        self.model.add(layers.Dense(7, activation=activations.softmax))
        self.lr_scheduler = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.00002,
            decay_steps=5750,
            decay_rate=0.3,
        )
        self.optimizer = optimizers.Adam(learning_rate=self.lr_scheduler)
        self.loss = losses.CategoricalCrossentropy()
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ['accuracy'],
        )
        self.model.summary(print_fn=self.summary_print)
        self.details_string()

    def summary_print(self, summary_line: str):
        # print(summary_line)
        self.description += summary_line + "\n"

    def details_string(self):
        for i, layer in enumerate(self.model.layers):
            self.description += "\n\n"
            self.description += str(layer) + "\n"
            for attr in dir(layer):
                try: 
                    if attr[0] != "_":
                        if attr not in ['trainable_variables', 'trainable_weights', 'weights', 'variables', 'kernel']:
                            self.description += attr + ": "
                            self.description += str(eval(f"self.model.layers[{i}].{attr}")) + "\n"
                except AttributeError:
                    pass
        self.description += "\n\n\n"
        self.description += "Optimizer:" + str(self.optimizer) + "\n"
        self.description += "Loss:" + str(self.loss) + "\n"
        self.description += "Learning Rate Scheduler:" + str(self.lr_scheduler) + "\n"
        for attr in dir(self.lr_scheduler):
            if attr[0] != "_" and not callable(getattr(self.lr_scheduler, attr)):
                self.description += attr + ": "
                self.description += str(eval(f"self.lr_scheduler.{attr}")) + "\n"

model = Model((395, 395, 3))

with open('description.txt', 'w') as description:
    print(model.description, file=description)

csv_logger = callbacks.CSVLogger('history.csv')

try:
    model.model.fit(
        train,
        batch_size=32,
        epochs=500,
        verbose=1,
        validation_data=valid,
        validation_batch_size=32,
        callbacks=csv_logger,
    )
except KeyboardInterrupt:
    save_stuff(model)
else:
    save_stuff(model)