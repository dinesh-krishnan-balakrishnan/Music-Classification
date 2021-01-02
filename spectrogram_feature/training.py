import tensorflow as TF
import os
import random

from spec_augment import initialize_augmentation

# -------------------- CONSTANTS -------------------- #

# TFRecord Directories
RECORD_DIR = os.path.join('D:', 'Track_Records')
TRAIN_DIR = os.path.join(RECORD_DIR, 'train')
TEST_DIR = os.path.join(RECORD_DIR, 'test')

# Training & Testing Dataset Sizes
N_TRAIN = 71_421
N_TEST = 7_714
N_RECORD = 1_000

# Spectrogram Constants
DIM = (430, 84)
SPEC_AUGMENT = None

# TFRecord Byte Formatting
FEATURE_STRUCTURE = {
    'parameters': TF.io.FixedLenFeature([DIM[0] * DIM[1]], TF.float32),
    'label': TF.io.FixedLenFeature([], TF.int64)
}

# -------------------- MAIN FUNCTION --------------------- #

def train_model(model, optimizer = TF.keras.optimizers.Adam(lr = 1e-3),
                batch_size = 64, epochs = 10, callbacks = None, augment = True):
    # Augmentation Initialization
    if augment:
        SPEC_AUGMENT = initialize_augmentation(DIM)
    
    # Dataset Generation
    TRAIN_data = generate_TRAIN(batch_size, epochs)
    TEST_data = generate_TEST(batch_size, epochs)
        
    # Compilation
    model.compile(
            optimizer,
            loss = 'sparse_categorical_crossentropy', 
            metrics = ['accuracy']
    )
        
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

# ----------------- DATASET GENERATION ----------------- #

def generate_TRAIN(batch_size, epochs):
    files = [os.path.join(TRAIN_DIR, file) for file in os.listdir(TRAIN_DIR)]
    
    # Sharded & Shuffled Dataset
    dataset = TF.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files))
    
    # Interleaved, Repeated, & Shuffled Dataset
    dataset = TF_interleave(dataset, file_dataset)
    dataset = dataset.repeat(epochs)
    dataset = dataset.shuffle(N_RECORD)
    
    # Parsed & Augmented Dataset
    dataset = TF_map(dataset, parse_dataset)
    dataset = TF_map(dataset, augment)
    dataset = dataset.apply(TF.data.experimental.ignore_errors())
    
    # Batched Dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(TF.data.experimental.AUTOTUNE)
    
    return dataset

def generate_TEST(batch_size, epochs):
    files = [os.path.join(TEST_DIR, file) for file in os.listdir(TEST_DIR)]
    
    # Interleaved & Repeated Dataset
    dataset = TF.data.Dataset.from_tensor_slices(files)
    dataset = TF_interleave(dataset, file_dataset)
    dataset = dataset.repeat(epochs)
    
    # Parsed Dataset
    dataset = TF_map(dataset, parse_dataset)
    
    # Batched Dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(TF.data.experimental.AUTOTUNE)
    
    return dataset
    
# ----------------- FUNCTION MAPPERS ----------------- #
    
def TF_interleave(dataset, map_func):
    return dataset.interleave(map_func,
              num_parallel_calls = TF.data.experimental.AUTOTUNE, deterministic = False)
                          
def TF_map(dataset, map_func):
    return dataset.map(map_func,
               num_parallel_calls = TF.data.experimental.AUTOTUNE, deterministic = False)
                          
# ----------------- MAPPING FUNCTIONS ----------------- #

# 
def file_dataset(file):
    return TF.data.TFRecordDataset(file)

def parse_dataset(feature):
    data = TF.io.parse_single_example(feature, FEATURE_STRUCTURE)
    data['parameters'] = TF.reshape(data['parameters'], DIM)
    return data['parameters'], data['label']

def augment(image, label):
    if SPEC_AUGMENT is not None:
        image = SPEC_AUGMENT(image)
    return image, label