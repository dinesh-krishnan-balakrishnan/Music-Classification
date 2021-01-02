import tensorflow as TF
import matplotlib.pyplot as PLT
import os

from spec_augment import initialize_augmentation

# -------------------- CONSTANTS -------------------- #

# Data File Location
RECORDS = 'D:/Spectrogram_Records'

# Training & Testing Dataset Size
N_TRAIN = 22356
N_TEST = 2490

# Image Normalization Value
NORMALIZE = (1. / 255)

# TFRecord Byte Formatting
FEATURE_STRUCTURE = {
    'image': TF.io.FixedLenFeature([], TF.string),
    'label': TF.io.FixedLenFeature([], TF.int64)
}

# -------------------- FUNCTIONS --------------------- #

# Trains a given model and returns the training results.
def train_model(model, spectrogram_type, size, batch_size = 32, LR = 1e-5, epochs = 10, 
        callbacks = None, window_size = None, window_shift = None, augment = True, cache = True):    
    # Compilation
    model.compile(
            TF.keras.optimizers.Adam(lr = LR), 
            loss = 'sparse_categorical_crossentropy', 
            metrics = ['accuracy']
    )
    
    # Retrieving Datasets
    TRAIN_data, TEST_data = get_datasets(spectrogram_type, size, batch_size, epochs, 
                                         window_size, window_shift, augment, cache)
        
    # Training
    history = model.fit(
        TRAIN_data,
        validation_data = TEST_data,
        epochs = epochs,
        steps_per_epoch = (N_TRAIN // batch_size),
        validation_steps = (N_TEST // batch_size),
        verbose = 1,
        callbacks = callbacks
    )
    
    # Returning the training results.
    return history.history

# Plots training results.
def plot_graphs(history, string):
    PLT.plot(history[string])
    PLT.plot(history['val_' + string])
    PLT.xlabel("Epochs")
    PLT.ylabel(string)
    PLT.legend([string, 'val_' + string])
    PLT.show()

# ----------------- DATASET GENERATION ----------------- #

# Retrieves the training & testing datasets based on specifications.
def get_datasets(spectrogram_type, image_size, batch_size, epochs, W_size, W_shift, augment, cache):
    # Computing the general file path.
    general_name = os.path.join(RECORDS, f'{spectrogram_type}_{image_size[0]}x{image_size[1]}')
    
    # Initializing image augmentation functions based on image size & model.
    W = (W_size is not None and W_shift is not None)
    spec_augment = initialize_augmentation(image_size, (not W))
    sliding_window = initialize_windowing(W, W_size, W_shift, image_size)

    # Training & Testing Data
    TRAIN_data = generate_dataset(True, general_name, batch_size, epochs, spec_augment, augment, sliding_window, cache)    
    TEST_data = generate_dataset(False, general_name, batch_size, epochs, spec_augment, augment, sliding_window, cache)
    
    return TRAIN_data, TEST_data

# Creates a TF.data.Dataset for efficient data processing.
def generate_dataset(training, general_name, batch_size, epochs, spec_augment, augment, sliding_window, cache):
    # Retrieving the dataset TFRecord.
    if training:
        dataset = TF.data.TFRecordDataset(f'{general_name}_train.tfr')
        dataset = dataset.shuffle(1_000) 
    else:
        dataset = TF.data.TFRecordDataset(f'{general_name}_test.tfr')

    # Retrieving & caching dataset bytes in memory.
    dataset = dataset_map(dataset, parse_data)
    if cache: dataset = dataset.cache()

    # Repeating dataset for multiple epochs of training & testing.
    dataset = dataset.repeat(epochs)

    # Image Preprocessing
    compute_image = initialize_image_computation(training, augment, spec_augment)
    dataset = dataset_map(dataset, compute_image)
    dataset = dataset.apply(TF.data.experimental.ignore_errors())
        
    # Windowing
    if sliding_window is not None:
        dataset = dataset_map(dataset, sliding_window)

    # Batching
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(TF.data.experimental.AUTOTUNE)
    
    return dataset

# ----------------- MAPPING FUNCTIONS ----------------- #

def dataset_map(dataset, function):
    return dataset.map(
        function,
        num_parallel_calls = TF.data.experimental.AUTOTUNE,
        deterministic = False
    )

# Byte Data Processor
def parse_data(feature):
    data = TF.io.parse_single_example(feature, FEATURE_STRUCTURE)
    return data['image'], data['label']

# Image Byte Decoder
def initialize_image_computation(training, augment, spec_augment):
    def compute_image(image_bytes, label):
        # Byte to Float Tensor Conversion
        image = TF.cast(TF.io.decode_png(image_bytes), TF.float32)

        # Data Augmentation
        image = (image * NORMALIZE - 0.5) # Normalization
        
        if training and augment: 
            image = spec_augment(image) # Spectrogram Augmentation
            
        image = TF.image.transpose(image) # Transpose

        return image, label
    
    return compute_image

# Creates a sliding window tensor based on the original image tensor.
def initialize_windowing(W, W_size, W_shift, image_size):
    def sliding_window(image, label):
        return (
            TF.stack([image[(end - W_size) : end] for end in range(W_size, image_size[0], W_shift)]),
            label
        )
    
    return sliding_window if W else None