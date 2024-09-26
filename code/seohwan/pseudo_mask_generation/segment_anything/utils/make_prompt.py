import cv2
import numpy as np

def make_box_prompt(mask: np.array, 
                    scale_factor: float = 1.0,
                    return_xyxy: bool = True) -> np.array:
    """
    Generate a box prompt that includes the mask(True region).

    Args:
        mask (np.array): given mask 
        scale_factor (float, optional): Adjust the size of the resulting box. 
                                        It is the same as cv2.boundingRect when set to 1.0. Defaults to 1.0.
        return_xyxy(bool, optional) : if True, return the coordinates in the form of (x1, y1, x2, y2).
                                      if Fale, return the coordinates in the form of (x, y, w, h). Defaults to True.
    Returns:
        np.array: coords of box, (x1, y1, x2, y2)
    """
    
    # mask shape 
    H, W = mask.shape[-2:]
    
    non_zero_points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(non_zero_points)
    
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    x = x - int((new_w - w) / 2)
    y = y - int((new_h - h) / 2)
    
    # if the new coords are outside the box 
    x = x if x > 0 else 0 
    y = y if y > 0 else 0 
    
    if x + new_w >= W:
        new_w = W - x - 1
    
    if y + new_h >= H:
        new_h = H - y - 1
    
    if return_xyxy:
        x1, y1, x2, y2 = x, y, x + new_w, y + new_h
        rect = np.array((x1, y1, x2, y2))
        
    else:
        rect = np.array((x, y, new_w, new_h))

    return rect


def make_whole_box_prompt(mask: np.array, 
                    scale_factor: float = 1.0,
                    return_xyxy: bool = True) -> np.array:
    """
    Generate a box prompt that includes the mask(True region).

    Args:
        mask (np.array): given mask 
        scale_factor (float, optional): Adjust the size of the resulting box. 
                                        It is the same as cv2.boundingRect when set to 1.0. Defaults to 1.0.
        return_xyxy(bool, optional) : if True, return the coordinates in the form of (x1, y1, x2, y2).
                                      if Fale, return the coordinates in the form of (x, y, w, h). Defaults to True.
    Returns:
        np.array: coords of box, (x1, y1, x2, y2)
    """
    
    # mask shape 
    H, W = mask.shape[-2:]
    
    non_zero_points = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(non_zero_points)
    
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    x = x - int((new_w - w) / 2)
    y = y - int((new_h - h) / 2)
    
    # if the new coords are outside the box 
    x = x if x > 0 else 0 
    y = y if y > 0 else 0 
    
    if x + new_w >= W:
        new_w = W - x - 1
    
    if y + new_h >= H:
        new_h = H - y - 1
    
    if return_xyxy:
        x1, y1, x2, y2 = x, y, x + new_w, y + new_h
        rect = np.array((0, 0, H, W))
        
    else:
        rect = np.array((x, y, new_w, new_h))

    return rect

def make_point_prompt(mask: np.array, 
                      n_point: int = 1) -> np.array:
    """
    Generate point prompts that belong inside the mask(True region).

    Args:
        mask (np.array): given mask. The mask must have values of either 0 or 1.
        n_point (int, optional): The number of point prompts to generate. Defaults to 1.

    Returns:
        np.array: list of tuple. Each tuple has coords of a point prompt, (x, y)
    """

    non_zero_points = cv2.findNonZero(mask)
    
    if non_zero_points is None or len(non_zero_points) == 0:
        print("No non-zero points found in the mask.")
        return np.array([])

    # Randomly select 'n_point' points from the list of non-zero points
    selected_points = non_zero_points[np.random.choice(non_zero_points.shape[0], n_point, replace=False)]

    # Convert to numpy array of tuples (x, y)
    return np.array([(point[0][0], point[0][1]) for point in selected_points])


def make_whole_box_random_point_prompt(mask: np.array,
                                       n_point: int = 1) -> np.array:
    """
    Generate random point prompts within the entire box defined by the mask dimensions.

    Args:
        mask (np.array): given mask. The mask must have values of either 0 or 1.
        n_point (int, optional): The number of point prompts to generate. Defaults to 1.

    Returns:
        np.array: list of tuple. Each tuple has coords of a point prompt, (x, y)
    """
    height, width = mask.shape

    # Generate random x, y coordinates within the dimensions of the mask
    random_x = np.random.randint(0, width, size=n_point)
    random_y = np.random.randint(0, height, size=n_point)

    # Combine x and y coordinates into a list of tuples
    random_points = np.array([(x, y) for x, y in zip(random_x, random_y)])

    return random_points


def make_point_prompt_half(mask: np.array, n_point: int = 1) -> np.array:
    """
    Generate point prompts inside and outside the mask region.

    Args:
        mask (np.array): Given mask. The mask must have values of either 0 or 1.
        n_point (int, optional): The total number of point prompts to generate. Defaults to 1.

    Returns:
        np.array: List of tuples. Each tuple has coordinates of a point prompt, (x, y)
    """
    if n_point <= 0:
        return np.array([])
    
    # Find non-zero and zero points
    non_zero_points = cv2.findNonZero(mask)
    zero_points = cv2.findNonZero(1 - mask)

    # Prepare the output array
    points = []

    # Half points from non-zero if available
    if non_zero_points is not None and len(non_zero_points) > 0:
        selected_non_zero_points = non_zero_points[np.random.choice(non_zero_points.shape[0], min(len(non_zero_points), n_point // 2), replace=False)]
        points.extend([(point[0][0], point[0][1]) for point in selected_non_zero_points])
    
    # Half points from zero if available
    if zero_points is not None and len(zero_points) > 0:
        selected_zero_points = zero_points[np.random.choice(zero_points.shape[0], min(len(zero_points), n_point - len(points)), replace=False)]
        points.extend([(point[0][0], point[0][1]) for point in selected_zero_points])

    return np.array(points)



def make_proba_point_prompt(softmax_cam: np.array, 
                      cam_mask: np.array, 
                      n_point: int = 1) -> np.array:
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
    sampled_indices = np.random.choice(len(valid_indices), size=min(n_point, len(valid_indices)), replace=False, p=valid_probs)
    
    random_points = valid_indices[sampled_indices]
    
    # Swap x and y coordinates
    random_points = np.fliplr(random_points)

    return np.ascontiguousarray(random_points)


def find_highest_proba_points(softmax_cam: np.array, cam_mask: np.array, n_point: int = 1) -> np.array:
    """
    Return the coordinates of the points with the highest probabilities in softmax_cam,
    restricted by cam_mask, in descending order of their probabilities.

    Args:
        softmax_cam (np.array): Array of probabilities, scaled between 0 and 1.
        cam_mask (np.array): Binary mask array with values of either 0 or 1.
        n_point (int, optional): Number of point coordinates to return. Defaults to 1.

    Returns:
        np.array: Array of tuples, each tuple has the coordinates of a point, (x, y)
    """
    # Apply the mask to the softmax_cam
    masked_probs = np.where(cam_mask == 1, softmax_cam, 0)
    
    # Flatten the array and get the indices of the top n_point probabilities
    flat_indices = np.argpartition(-masked_probs.flatten(), n_point)[:n_point]
    
    # Convert flat indices to 2D indices
    top_indices = np.unravel_index(flat_indices, softmax_cam.shape)
    
    # Swap x and y coordinates
    points = np.column_stack((top_indices[1], top_indices[0]))
    
    # Sort points by their probabilities in descending order
    points = points[np.argsort(-softmax_cam[top_indices])]
    
    return points






