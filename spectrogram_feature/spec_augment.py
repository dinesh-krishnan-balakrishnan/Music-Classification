import tensorflow as TF
from tensorflow_addons.image import sparse_image_warp

# Initializing augmentation parameters using function scope.
def initialize_augmentation(image_size, apply_mask = True):
    # Width & Height
    W = TF.constant(image_size[0], dtype = TF.int32)
    H = TF.constant(image_size[1], dtype = TF.int32)
    
    # Time Warping & Masking Constraints
    MAX_SHIFT = (W // 10)
    MAX_MASK = 0.1
    APPLY_MASK = apply_mask

    # ------------------- MAIN FUNCTION ------------------- #
    
    # Returns a warped & masked image.
    def spectrogram_augmentation(image):
        image = time_warp(image)

        if APPLY_MASK: 
            image = mask_image(image)

        return image

    # ------------------- TIME WARPING ------------------- #

    def time_warp(image):
        # Generating the start & end positions for warping.
        warp_start = TF.random.uniform([], 
                       minval = MAX_SHIFT, maxval = (W - MAX_SHIFT), dtype = TF.int32)
        warp_end = TF.random.uniform([], 
                     minval = -MAX_SHIFT, maxval = MAX_SHIFT, dtype = TF.int32) + warp_start

        # Generating the control points depicting the before & after states of the image.
        source = get_points(warp_start)
        destination = get_points(warp_end)

        # Adding dimensions to meet the requirements of the sparse_image_warp function.
        image = TF.expand_dims(image, 0)
        image = TF.expand_dims(image, 3)
        source = TF.expand_dims(source, 0)
        destination = TF.expand_dims(destination, 0)

        # Generating & Returning Warped Image
        warped_image, _ = sparse_image_warp(image, source, destination)
        return TF.squeeze(warped_image)

    # Generates control points depicting the state of the image.
    def get_points(time):
        return TF.convert_to_tensor([
            [0,        0       ],
            [(W - 1),  0       ],
            [0,        (H - 1) ],
            [(W - 1),  (H - 1) ],
            [time,     0       ],
            [time,     (H - 1) ],
            [time,     (H // 2)]
        ], dtype = TF.float32)

    # --------------------- MASKING --------------------- #

    def mask_image(image):        
        # Applying 2 Frequency (W) & 2 Time Series (H) Masks
        image -= 0.5
        image *= W_mask() * W_mask() * H_mask() * H_mask()
        image += 0.5
        
        return image

    # Time Series Mask
    def W_mask():
        W_size, W_position = generate_blank(W)
        return TF.concat((
            TF.ones( (W_position,              H)),
            TF.zeros((W_size,                  H)),
            TF.ones( (W - W_position - W_size, H))
        ), 0)
    
    # Frequency Mask
    def H_mask():
        H_size, H_position = generate_blank(H)
        return TF.concat((
            TF.ones( (W, H_position,            )),
            TF.zeros((W, H_size,                )),
            TF.ones( (W, H - H_position - H_size))
        ), 1)

    # Generating augmentation size & position.
    def generate_blank(N):
        N_size = TF.cast(
            TF.random.uniform([], maxval = MAX_MASK, dtype = TF.float32) * TF.cast(N, TF.float32), 
            TF.int32
        )
        N_position = TF.random.uniform([], maxval = (N - N_size), dtype = TF.int32)
        return N_size, N_position
    
    # --------------------------------------------------- #
    return spectrogram_augmentation