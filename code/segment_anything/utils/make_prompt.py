import cv2
import numpy as np

def make_point_prompt(
    mask: np.array, 
    n_point: int = 1
) -> np.array:
    """
    Generate point prompts that belong inside the mask (True region).

    Args:
        mask (np.array): given mask. The mask must have values of either 0 or 1.
        n_point (int, optional): The number of point prompts to generate. Defaults to 1.

    Returns:
        np.array: list of tuple. Each tuple has coords of a point prompt, (x, y)
    """

    non_zero_points = cv2.findNonZero(mask)
    
    if non_zero_points is None or len(non_zero_points) == 0:
        print("No non-zero points found in the mask.")
        return np.array([[0, 0]])

    # Randomly select 'n_point' points from the list of non-zero points
    selected_points = non_zero_points[np.random.choice(non_zero_points.shape[0], n_point, replace=False)]

    # Convert to numpy array of tuples (x, y)
    return np.array([(point[0][0], point[0][1]) for point in selected_points])


def make_proba_point_prompt(
    softmax_cam: np.array, 
    cam_mask: np.array, 
    n_point: int = 1
) -> np.array:
    """
    Generate point prompts based on softmax_cam probabilities within the cam_mask region.

    Args:
        softmax_cam (np.array): Softmax CAM with probability values between 0 and 1.
        cam_mask (np.array): Binary mask with values of either 0 or 1.
        n_point (int, optional): The number of point prompts to generate. Defaults to 1.

    Returns:
        np.array: List of tuples. Each tuple has coords of a point prompt, (x, y)
    """
    # Ensure cam_mask is binary
    cam_mask = (cam_mask > 0).astype(np.uint8)
    
    # Get probabilities for valid points
    valid_probs = softmax_cam[cam_mask == 1]
    
    # Normalize probabilities
    valid_probs = valid_probs / np.sum(valid_probs)
    
    # Get indices of valid points
    valid_indices = np.column_stack(np.where(cam_mask == 1))
    
    # Sample points based on probabilities
    try:
        sampled_indices = np.random.choice(len(valid_indices), size=min(n_point, len(valid_indices)), replace=False, p=valid_probs)
        random_points = valid_indices[sampled_indices]
        
    except:
        random_points = np.array([[0, 0]])
        
    # Swap x and y coordinates
    random_points = np.fliplr(random_points)

    return np.ascontiguousarray(random_points)