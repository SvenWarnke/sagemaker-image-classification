import argparse, os
import numpy as np

import tensorflow as tf
import keras
from keras import backend as K
from keras import models
from keras import layers
from keras.optimizers import SGD
from keras.utils import multi_gpu_model

from tensorflow.keras import applications


def get_model(image_shape):
    inputs = layers.Input(shape=(*image_shape, 3))
    
    Net = applications.inception_v3.InceptionV3
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
        directory / 'train',  # this is the target directory
        target_size=image_shape,  # all images will be resized 
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator


def get_validation_generator(directory, image_shape, batch_size):
    test_datagen = image.ImageDataGenerator(rescale=1. / 255)

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        directory / 'test',
        target_size=image_shape,
        batch_size=batch_size,
        class_mode='categorical')
    return validation_generator


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
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
    
#     x_train = np.load(os.path.join(training_dir, 'training.npz'))['image']
#     y_train = np.load(os.path.join(training_dir, 'training.npz'))['label']
#     x_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
#     y_val  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']
    
    

#     # Tensorflow needs image channels last, e.g. (batch size, width, height, channels)
#     K.set_image_data_format('channels_last')  
#     print(K.image_data_format())

#     if K.image_data_format() == 'channels_first':
#         print("Incorrect configuration: Tensorflow needs channels_last")
#     else:
#         # channels last
#         x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#         x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
#         input_shape = (img_rows, img_cols, 1)
#         batch_norm_axis=-1

#     print('x_train shape:', x_train.shape)
#     print(x_train.shape[0], 'train samples')
#     print(x_val.shape[0], 'test samples')
    
#     # Normalize pixel values
#     x_train  = x_train.astype('float32')
#     x_val    = x_val.astype('float32')
#     x_train /= 255
#     x_val   /= 255
    
#     # Convert class vectors to binary class matrices
#     num_classes = 10
#     y_train = keras.utils.to_categorical(y_train, num_classes)
#     y_val   = keras.utils.to_categorical(y_val, num_classes)
    
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
    
#     model.fit(x_train, y_train, batch_size=batch_size,
#                   validation_data=(x_val, y_val), 
#                   epochs=epochs,
#                   verbose=1)
    
#     score = model.evaluate(x_val, y_val, verbose=0)
#     print('Validation loss    :', score[0])
#     print('Validation accuracy:', score[1])
    
    save_path = model_dir + '/model'
    
    if not os.path.exists(save_path):
        print('save directories...', flush=True)
        os.makedirs(save_path)
    
    model.save(save_path + '/mymodel.h5')