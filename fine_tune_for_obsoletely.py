import argparse, os
import numpy as np
import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications


from subprocess import call
call("pip install efficientnet==1.1.1".split(" "))
import efficientnet.tfkeras as efn

NETS = {
    "EfficientNetB0": efn.EfficientNetB0,
    "EfficientNetB5": efn.EfficientNetB5,
    "InceptionV3": applications.InceptionV3,
    "MobileNetV2": applications.MobileNetV2,
    "ResNet50": applications.ResNet50,
}


def get_model(Net, image_shape):
    inputs = layers.Input(shape=(*image_shape, 3))
    
    base_efficient_net = Net(weights='imagenet', input_tensor=inputs, include_top=False)

    base_efficient_net.trainable = False

    x = base_efficient_net.output
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    predictions = layers.Dense(5, activation='softmax')(x)

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
    parser.add_argument('--log-dir', type=str)
    parser.add_argument('--steps-per-epoch', type=int, default=10)
    parser.add_argument('--model', type=str)
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    training_dir   = args.training
    validation_dir = args.validation
    log_dir = args.log_dir
    steps_per_epoch = args.steps_per_epoch
    Net = NETS[args.model]
    
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
    
    model = get_model(
        image_shape=image_shape,
        Net=Net
    )
    
    print(model.summary())

    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
    
    
    tensorboard_cb = callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
    )
        
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        verbose=0, 
        mode='min'
    )
    
    checkpoint_path = 'model.h5'
    
    checkpoint_cb = callbacks.ModelCheckpoint(
        checkpoint_path, 
        save_best_only=True, 
        monitor='val_acc'  # for tensorflow 2 change to val_accuracy
    )
    
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=1,
        callbacks=[
            tensorboard_cb,
            early_stopping_cb,
            checkpoint_cb
        ]
    )
    
    tensorboard_cb_fine_tune = callbacks.TensorBoard(
        log_dir=log_dir + "_tune",
        histogram_freq=1,
    )
    
    new_model = models.load_model(checkpoint_path)
    new_model.trainable = True
    
    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),  # Low learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    new_model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=1,
        callbacks=[
            tensorboard_cb_fine_tune,
            # early_stopping_cb,  # causes problems
        ]
    )
    
    
    save_path = model_dir + '/model'
    
    if not os.path.exists(save_path):
        print('save directories...', flush=True)
        os.makedirs(save_path)
    
    model.save(save_path + '/mymodel.h5')
