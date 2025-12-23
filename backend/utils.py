import cv2
import numpy as np

def preprocess_inputs(x):
    """
    Preprocess input images for xView2 models
    
    This function normalizes the input image using ImageNet statistics
    and converts from BGR to RGB format.
    
    Args:
        x: Input image in BGR format (OpenCV format)
        
    Returns:
        Preprocessed image as float32 array
    """
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    
    x = x.astype(np.float32)
    
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    
    x = (x - mean) / std
    
    return x


def preprocess_image(image_path):
    """
    Load and preprocess image from file path
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    return preprocess_inputs(img)


def preprocess_image_from_bytes(image_bytes):
    """
    Load and preprocess image from byte array
    
    Args:
        image_bytes: Image as byte array
        
    Returns:
        Preprocessed image
    """
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from bytes")
    return preprocess_inputs(img)


def postprocess_mask(mask, threshold=0.5):
    """
    Convert continuous mask predictions to binary
    
    Args:
        mask: Continuous prediction mask (0-1)
        threshold: Threshold for binarization
        
    Returns:
        Binary mask
    """
    return (mask > threshold).astype(np.uint8)


def resize_image(image, target_size=1024):
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: Target size for longest dimension
        
    Returns:
        Resized image and scale factor
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= target_size:
        return image, 1.0
    
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scale = target_size / max(h, w)
    
    return resized, scale


def pad_image(image, target_size=1024):
    """
    Pad image to square size
    
    Args:
        image: Input image
        target_size: Target square size
        
    Returns:
        Padded image and padding info
    """
    h, w = image.shape[:2]
    
    if h == target_size and w == target_size:
        return image, (0, 0, 0, 0)

    pad_h = target_size - h
    pad_w = target_size - w
    
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    if len(image.shape) == 3:
        padded = cv2.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
    else:
        padded = cv2.copyMakeBorder(
            image,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0
        )
    
    return padded, (pad_top, pad_bottom, pad_left, pad_right)


def unpad_image(image, padding):
    """
    Remove padding from image
    
    Args:
        image: Padded image
        padding: Padding info (top, bottom, left, right)
        
    Returns:
        Unpadded image
    """
    pad_top, pad_bottom, pad_left, pad_right = padding
    
    h, w = image.shape[:2]
    
    return image[
        pad_top:h-pad_bottom,
        pad_left:w-pad_right
    ]


def normalize_image(image):
    """
    Normalize image to 0-1 range
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image


def denormalize_image(image):
    """
    Denormalize image from 0-1 to 0-255 range
    
    Args:
        image: Normalized image
        
    Returns:
        Denormalized image as uint8
    """
    return (image * 255).astype(np.uint8)
