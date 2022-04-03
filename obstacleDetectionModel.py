import itertools
import os

import matplotlib.pylab as plt
import numpy as np
from PIL import Image
import glob

import tensorflow as tf
import tensorflow_hub as hub
!pip install pyyaml h5py  # Required to save models in HDF5 format


print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
model_name = "efficientnetv2-xl-21k" # @param ['efficientnetv2-s', 'efficientnetv2-m', 'efficientnetv2-l', 'efficientnetv2-s-21k', 'efficientnetv2-m-21k', 'efficientnetv2-l-21k', 'efficientnetv2-xl-21k', 'efficientnetv2-b0-21k', 'efficientnetv2-b1-21k', 'efficientnetv2-b2-21k', 'efficientnetv2-b3-21k', 'efficientnetv2-s-21k-ft1k', 'efficientnetv2-m-21k-ft1k', 'efficientnetv2-l-21k-ft1k', 'efficientnetv2-xl-21k-ft1k', 'efficientnetv2-b0-21k-ft1k', 'efficientnetv2-b1-21k-ft1k', 'efficientnetv2-b2-21k-ft1k', 'efficientnetv2-b3-21k-ft1k', 'efficientnetv2-b0', 'efficientnetv2-b1', 'efficientnetv2-b2', 'efficientnetv2-b3', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'bit_s-r50x1', 'inception_v3', 'inception_resnet_v2', 'resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152', 'resnet_v2_50', 'resnet_v2_101', 'resnet_v2_152', 'nasnet_large', 'nasnet_mobile', 'pnasnet_large', 'mobilenet_v2_100_224', 'mobilenet_v2_130_224', 'mobilenet_v2_140_224', 'mobilenet_v3_small_100_224', 'mobilenet_v3_small_075_224', 'mobilenet_v3_large_100_224', 'mobilenet_v3_large_075_224']

model_handle =  "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_xl/feature_vector/2"


#model_handle = model_handle_map.get(model_name)
pixels = 512

print(f"Selected model: {model_name} : {model_handle}")

IMAGE_SIZE = (pixels, pixels)
print(f"Input size {IMAGE_SIZE}")

BATCH_SIZE = 16 #@param {type:"integer"}

data_dir = "/content/trainingData"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode="categorical",
    # labels=data_labels,
    labels="inferred",  # can be infered or can be set in data_labels
    # class_names=["trafficlight", "stop", "speedlimit", "crosswalk"], #only necessary if data labels are inferred
    # color_mode='rgb',
    batch_size=1,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=123,
    subset="training",
    validation_split=.20,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    label_mode="categorical",
    # labels=data_labels,
    labels="inferred",
    # class_names=["trafficlight", "stop", "speedlimit", "crosswalk"],
    # color_mode='rgb',
    batch_size=1,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=123,
    subset="validation",
    validation_split=.20,
)
print(train_ds.class_names)

def build_dataset(subset):
  return tf.keras.preprocessing.image_dataset_from_directory(
      data_dir,
      validation_split=.20,
      subset=subset,
      label_mode="categorical",
      # Seed needs to provided when using validation_split and shuffle = True.
      # A fixed seed is used so that the validation set is stable across runs.
      seed=123,
      image_size=IMAGE_SIZE,
      batch_size=1)

#train_ds = build_dataset("training")
class_names = tuple(train_ds.class_names)
#class_names = tuple(["trafficlight", "stop", "speedlimit", "crosswalk"])
#class_names = tuple(["0", "1"])
#train_size = train_ds.cardinality().numpy()
#rain_size = 964   #This number changes based on how many numbers are in the training set
train_size = 1399
train_ds = train_ds.unbatch().batch(BATCH_SIZE)
train_ds = train_ds.repeat()

normalization_layer = tf.keras.layers.Rescaling(1. / 255)
preprocessing_model = tf.keras.Sequential([normalization_layer])
do_data_augmentation = False #@param {type:"boolean"}
if do_data_augmentation:
  preprocessing_model.add(
      tf.keras.layers.RandomRotation(40))
  preprocessing_model.add(
      tf.keras.layers.RandomTranslation(0, 0.2))
  preprocessing_model.add(
      tf.keras.layers.RandomTranslation(0.2, 0))
  # Like the old tf.keras.preprocessing.image.ImageDataGenerator(),
  # image sizes are fixed when reading, and then a random zoom is applied.
  # If all training inputs are larger than image_size, one could also use
  # RandomCrop with a batch size of 1 and rebatch later.
  preprocessing_model.add(
      tf.keras.layers.RandomZoom(0.2, 0.2))
  preprocessing_model.add(
      tf.keras.layers.RandomFlip(mode="horizontal"))
train_ds = train_ds.map(lambda images, labels:
                        (preprocessing_model(images), labels))

#val_ds = build_dataset("validation")
#valid_size = val_ds.cardinality().numpy()
#valid_size = 240 #This value depends on how many images are in the valdiation set
valid_size = 349
val_ds = val_ds.unbatch().batch(BATCH_SIZE)
val_ds = val_ds.map(lambda images, labels:
                    (normalization_layer(images), labels))

# Recreate the exact same model, including its weights and the optimizer
path = '/content/gdrive/MyDrive/Colab Notebooks/my_model.h5'
new_model = tf.keras.models.load_model((path),custom_objects={'KerasLayer':hub.KerasLayer})

# Show the model architecture
new_model.summary()

x, y = next(iter(val_ds))
image = x[0, :, :, :]
true_index = np.argmax(y[0])
plt.imshow(image)
plt.axis('off')
plt.show()

'''
current_dir = os.getcwd()
print(current_dir)
#img= Image.open("../content/human_detection_dataset/human_detection_dataset/0/"114.png"")
#image_dir = "../content/human_detection_dataset/human_detection_dataset/0/"114.png""
image_list = []
for filename in glob.glob('human_detection_dataset/human_detection_dataset/0/*.png'): 
    im=Image.open(filename)
    image_list.append(im)
img= Image.open(image_list[0])
plt.imshow(img)
plt.axis('off')
plt.show()
'''

# Expand the validation image to (1, 224, 224, 3) before predicting the label
prediction_scores = new_model.predict(np.expand_dims(image, axis=0))
predicted_index = np.argmax(prediction_scores)
print("True label: " + class_names[true_index])
print("Predicted label: " + class_names[predicted_index])
