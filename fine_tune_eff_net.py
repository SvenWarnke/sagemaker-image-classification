import argparse, os
import numpy as np

import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import layers
#from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import multi_gpu_model

from subprocess import call
call("pip install efficientnet".split(" "))
call("pip install keras".split(" "))


import efficientnet.keras as efn
#from tensorflow.keras import applications
from tensorflow.keras.preprocessing import image




def get_model(image_shape):
    inputs = layers.Input(shape=(*image_shape, 3))
    
    Net = efn.EfficientNetB0
    base_efficient_net = Net(weights='imagenet', input_tensor=inputs, include_top=False)

    base_efficient_net.trainable = False

    x = base_efficient_net.output
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    predictions = layers.Dense(2, activation='softmax')(x)

    efficient_net = models.Model(inputs=base_efficient_net.input, outputs=predictions)

    efficient_net.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return efficient_net


def get_train_generator(directory, image_shape, batch_size):
    train_datagen = image.ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        directory,  # this is the target directory
        target_size=image_shape,  # all images will be resized 
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator


def get_validation_generator(directory, image_shape, batch_size):
    test_datagen = image.ImageDataGenerator(rescale=1. / 255)

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        directory,
        target_size=image_shape,
        batch_size=batch_size,
        class_mode='categorical')
    return validation_generator


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    
    # input image dimensions
    image_shape = (224, 224)
    
    
    
    train_generator = get_train_generator(
        directory=training_dir,
        image_shape=image_shape,
        batch_size=batch_size
    )
    
    validation_generator = get_validation_generator(
        directory=validation_dir,
        image_shape=image_shape,
        batch_size=batch_size
    )
    
    model = get_model(image_shape=image_shape)
    
    print(model.summary())

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
    
    
    model.fit(
        train_generator,
        steps_per_epoch=1,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=1,
    )
    
    save_path = model_dir + '/model'
    
    if not os.path.exists(save_path):
        print('save directories...', flush=True)
        os.makedirs(save_path)
    
    model.save(save_path + '/mymodel.h5')