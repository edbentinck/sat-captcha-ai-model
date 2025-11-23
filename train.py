import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

# Optimize thread configuration for Apple Silicon M4 Pro
# Setting to 0 allows TensorFlow to dynamically use all available cores
# This is recommended for Apple Silicon which has performance + efficiency cores
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

# Enable mixed precision training for faster training on Apple Silicon
# This uses float16 for computations while keeping float32 for stability
# Note: For small models/datasets, mixed precision overhead may outweigh benefits
# Set to False to disable and test performance
ENABLE_MIXED_PRECISION = True
if ENABLE_MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print("Mixed precision enabled")
else:
    print("Mixed precision disabled (using float32)")

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from model import train_model
from configs import ModelConfigs

import os
import numpy as np
import csv
import cv2

# Create a list of all the images and labels in the dataset
dataset, vocab, max_len = [], set(), 0
captcha_path = os.path.join("dataset")
for file in os.listdir(captcha_path):
    file_path = os.path.join(captcha_path, file)
    label = os.path.splitext(file)[0] # Get the file name without the extension
    dataset.append([file_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))

configs = ModelConfigs()

# Save vocab and maximum text length to configs
configs.vocab = "".join(vocab)
configs.max_text_length = max_len

# Optimize batch size for small dataset (~376 images)
# For small datasets, smaller batches (64) generalize better and train faster
# Only increase batch size if dataset is large enough to benefit
original_batch_size = configs.batch_size  # Save original (64)
train_size = int(len(dataset) * 0.9)  # Training set size

# Use original batch size (64) for optimal performance on small datasets
# Only increase if we have enough data (e.g., >1000 samples)
if train_size > 1000:
    configs.batch_size = min(128, train_size // 8)  # More conservative for larger datasets
else:
    configs.batch_size = original_batch_size  # Keep 64 for small datasets

print(f"Using batch size: {configs.batch_size} (original: {original_batch_size}, train samples: {train_size})")

# Scale learning rate only if batch size increased significantly
batch_size_ratio = configs.batch_size / original_batch_size
if batch_size_ratio > 1.2:  # Only scale if batch size increased by >20%
    # Square root scaling is conservative
    configs.learning_rate = configs.learning_rate * (batch_size_ratio ** 0.5)
    print(f"Scaled learning rate to {configs.learning_rate:.6f} (batch size ratio: {batch_size_ratio:.2f}x)")
else:
    print(f"Keeping original learning rate: {configs.learning_rate:.6f}")

configs.save()

# Split dataset into training and validation sets (90% train, 10% val)
split_idx = int(len(dataset) * 0.9)
train_dataset = dataset[:split_idx]
val_dataset = dataset[split_idx:]

print(f"Loading {len(train_dataset)} training images and {len(val_dataset)} validation images into memory...")

# Create vocab mapping
vocab_list = sorted(list(vocab))
char_to_num = {char: idx for idx, char in enumerate(vocab_list)}
num_to_char = {idx: char for idx, char in enumerate(vocab_list)}
padding_value = len(vocab_list)

# Pre-load all images into memory once (much faster than reading from disk each epoch)
# For ~376 images at 60x160x3, this is only ~43MB - very manageable
# Use simpler approach to avoid tf.data overhead and warnings
def preload_images(dataset_list, width, height):
    """Pre-load and resize all images into memory efficiently"""

    images = []
    labels = []

    # Use OpenCV for faster image loading (more efficient than tf.io for pre-loading)
    for file_path, label in dataset_list:
        # Load with OpenCV (BGR format)
        img = cv2.imread(file_path)
        if img is None:
            print(f"Warning: Could not load {file_path}")
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)

        # Convert to float32 (0-255 range, model will normalize)
        img = img.astype(np.float32)

        images.append(img)
        labels.append(label)

    return np.array(images, dtype=np.float32), labels

print("Pre-loading training images...")
train_images, train_labels = preload_images(train_dataset, configs.width, configs.height)
print("Pre-loading validation images...")
val_images, val_labels = preload_images(val_dataset, configs.width, configs.height)
print("Images loaded into memory!")

# Pre-encode labels to avoid py_function overhead during training
def encode_labels(labels, vocab_dict, max_length, padding_value):
    """Pre-encode all labels to integers"""
    encoded_labels = []
    for label in labels:
        encoded = np.array([vocab_dict.get(c, 0) for c in label], dtype=np.int32)
        # Pad to max_length
        if len(encoded) < max_length:
            padding = np.full(max_length - len(encoded), padding_value, dtype=np.int32)
            encoded = np.concatenate([encoded, padding])
        else:
            encoded = encoded[:max_length]
        encoded_labels.append(encoded)
    return np.array(encoded_labels, dtype=np.int32)

print("Pre-encoding labels...")
train_labels_encoded = encode_labels(train_labels, char_to_num, configs.max_text_length, padding_value)
val_labels_encoded = encode_labels(val_labels, char_to_num, configs.max_text_length, padding_value)
print("Labels encoded!")

def augment_batch(images):
    """Apply vectorized augmentation optimized for CAPTCHA images (much faster than tf.map_fn)"""
    # Random brightness (small variation - CAPTCHAs have controlled lighting)
    # Vectorized: applies to entire batch at once
    brightness_delta = tf.random.uniform([tf.shape(images)[0]], -0.15, 0.15)
    brightness_delta = tf.reshape(brightness_delta, [-1, 1, 1, 1])
    images = images + brightness_delta
    images = tf.clip_by_value(images, 0.0, 1.0)

    # Random contrast (simulates slight compression artifacts)
    # Vectorized: applies to entire batch at once
    contrast_factor = tf.random.uniform([tf.shape(images)[0]], 0.85, 1.15)
    contrast_factor = tf.reshape(contrast_factor, [-1, 1, 1, 1])
    mean = tf.reduce_mean(images, axis=[1, 2, 3], keepdims=True)
    images = (images - mean) * contrast_factor + mean
    images = tf.clip_by_value(images, 0.0, 1.0)

    # NOTE: Removed rotation - CAPTCHA text should never be rotated!
    # NOTE: Removed erode/dilate - not needed for basic CAPTCHA augmentation
    return images

def preprocess_batch(images, labels, augment=False):
    """Process batch of pre-loaded images"""
    images = tf.cast(images, tf.float32)

    # Apply vectorized augmentation if training (much faster than tf.map_fn)
    if augment:
        images = augment_batch(images)

    labels = tf.cast(labels, tf.int32)
    return images, labels

def create_tf_dataset(images, labels, batch_size, augment=False, shuffle=True):
    """Create optimized tf.data.Dataset from pre-loaded images"""
    # Create dataset from pre-loaded numpy arrays (much faster!)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    # Shuffle if training
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images), reshuffle_each_iteration=True)

    # Batch first (more efficient with pre-loaded data)
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # Apply augmentation if training (vectorized, no map needed for validation)
    if augment:
        dataset = dataset.map(
            lambda imgs, lbls: preprocess_batch(imgs, lbls, augment=True),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )
    else:
        # For validation, just cast types (no augmentation needed)
        dataset = dataset.map(
            lambda imgs, lbls: (tf.cast(imgs, tf.float32), tf.cast(lbls, tf.int32)),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True
        )

    # Prefetch for better performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Create tf.data datasets from pre-loaded images (much faster!)
print("Creating optimized tf.data pipelines...")
train_tf_dataset = create_tf_dataset(
    train_images,
    train_labels_encoded,
    batch_size=configs.batch_size,
    augment=True,  # Apply augmentation for training
    shuffle=True
)

val_tf_dataset = create_tf_dataset(
    val_images,
    val_labels_encoded,
    batch_size=configs.batch_size,
    augment=False,  # No augmentation for validation
    shuffle=False
)
print("Data pipelines ready!")

# Create generator wrappers for compatibility with model.fit()
# Note: tf.data.Dataset can be used directly with model.fit(), but we wrap it as a generator
# for compatibility with any custom callbacks that might expect iterables
def dataset_generator(dataset):
    """Generator wrapper for tf.data.Dataset"""
    for batch in dataset:
        yield batch

# Use tf.data.Dataset directly (more efficient) or generator wrapper
train_data_provider = train_tf_dataset
val_data_provider = val_tf_dataset

# Creating TensorFlow model architecture
model = train_model(
    input_dim = (configs.height, configs.width, 3),
    output_dim = len(configs.vocab),
)

# Model Lambda layer handles normalization (x / 255)
# Images are kept in 0-255 range, Lambda normalizes to 0-1 during forward pass

# Ensure output layer uses float32 for mixed precision (softmax needs float32)
# Mixed precision will use float16 for intermediate layers but float32 for output
if ENABLE_MIXED_PRECISION and hasattr(model.layers[-1], 'dtype_policy'):
    model.layers[-1].dtype_policy = 'float32'

# Compile the model with mixed precision support (if enabled)
# Wrap optimizer with LossScaleOptimizer to prevent underflow in float16
# Learning rate has been scaled based on batch size for optimal convergence
if ENABLE_MIXED_PRECISION:
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(learning_rate=configs.learning_rate)
    )
else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=configs.learning_rate)

model.compile(
    optimizer=optimizer,
    loss=CTCloss(),
    metrics=[CWERMetric(padding_token=len(configs.vocab))],
    run_eagerly=False
)
model.summary(line_length=110)
# Define path to save the model
os.makedirs(configs.model_path, exist_ok=True)

# Define callbacks
earlystopper = EarlyStopping(monitor="val_CER", patience=50, verbose=1, mode="min")
checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True, mode="min")
trainLogger = TrainLogger(configs.model_path)
# Optimize TensorBoard: update every 10 batches instead of every batch (reduces I/O overhead)
tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=10)
reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=20, verbose=1, mode="min")
model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

# Train the model
model.fit(
    train_data_provider,
    validation_data=val_data_provider,
    epochs=configs.train_epochs,
    callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx]
)

# Save training and validation datasets as csv files
train_csv_data = [[item[0], item[1]] for item in train_dataset]
val_csv_data = [[item[0], item[1]] for item in val_dataset]

with open(os.path.join(configs.model_path, "train.csv"), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_path', 'label'])
    writer.writerows(train_csv_data)

with open(os.path.join(configs.model_path, "val.csv"), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['file_path', 'label'])
    writer.writerows(val_csv_data)
