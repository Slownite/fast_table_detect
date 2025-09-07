import pytest
import numpy as np
import cv2 as cv
from pathlib import Path

@pytest.fixture
def sample_image():
    """Create a simple test image with basic table-like structure."""
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Draw horizontal lines
    cv.line(img, (50, 50), (350, 50), (0, 0, 0), 2)
    cv.line(img, (50, 100), (350, 100), (0, 0, 0), 2)
    cv.line(img, (50, 150), (350, 150), (0, 0, 0), 2)
    cv.line(img, (50, 200), (350, 200), (0, 0, 0), 2)
    
    # Draw vertical lines
    cv.line(img, (50, 50), (50, 200), (0, 0, 0), 2)
    cv.line(img, (150, 50), (150, 200), (0, 0, 0), 2)
    cv.line(img, (250, 50), (250, 200), (0, 0, 0), 2)
    cv.line(img, (350, 50), (350, 200), (0, 0, 0), 2)
    
    # Add some text content inside cells
    cv.putText(img, "Cell 1", (60, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv.putText(img, "Cell 2", (160, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv.putText(img, "Cell 3", (260, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img

@pytest.fixture
def grayscale_image():
    """Create a grayscale test image."""
    img = np.ones((200, 300), dtype=np.uint8) * 255
    
    # Add some horizontal lines
    img[50:52, 20:280] = 0
    img[100:102, 20:280] = 0
    img[150:152, 20:280] = 0
    
    # Add some vertical lines
    img[50:150, 50:52] = 0
    img[50:150, 150:152] = 0
    img[50:150, 250:252] = 0
    
    return img

@pytest.fixture
def binary_image():
    """Create a binary test image."""
    img = np.zeros((100, 150), dtype=np.uint8)
    
    # Add some structure
    img[25:30, 25:125] = 255  # Horizontal line
    img[70:75, 25:125] = 255  # Another horizontal line
    img[25:75, 25:30] = 255   # Vertical line
    img[25:75, 75:80] = 255   # Another vertical line
    img[25:75, 120:125] = 255 # Third vertical line
    
    return img

@pytest.fixture
def noisy_image():
    """Create an image with noise for testing preprocessing."""
    img = np.random.randint(240, 255, (150, 200, 3), dtype=np.uint8)
    
    # Add some clear table structure
    cv.rectangle(img, (25, 25), (175, 125), (0, 0, 0), 2)
    cv.line(img, (25, 75), (175, 75), (0, 0, 0), 2)
    cv.line(img, (100, 25), (100, 125), (0, 0, 0), 2)
    
    return img

@pytest.fixture
def empty_image():
    """Create an empty image for negative tests."""
    return np.ones((100, 100, 3), dtype=np.uint8) * 255

@pytest.fixture
def rotated_image():
    """Create a slightly rotated table image for deskewing tests."""
    img = np.ones((200, 300, 3), dtype=np.uint8) * 255
    
    # Create a table
    cv.rectangle(img, (50, 50), (250, 150), (0, 0, 0), 2)
    cv.line(img, (50, 100), (250, 100), (0, 0, 0), 2)
    cv.line(img, (150, 50), (150, 150), (0, 0, 0), 2)
    
    # Rotate the image slightly
    center = (150, 100)
    rotation_matrix = cv.getRotationMatrix2D(center, 2, 1.0)
    rotated = cv.warpAffine(img, rotation_matrix, (300, 200))
    
    return rotated

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path / "test_data"