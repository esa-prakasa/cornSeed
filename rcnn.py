import numpy as np

# imagine a 3*3 image with a diagional line across
X = np.eye(3,3, dtype=np.uint8)
Y = np.eye(3,3, dtype=np.uint8)

# we change one pixel
X[1,1] = 0

def dice_coefficient(X, y):
    
    # convert the pixel/mask matrix to a one-dimensional series
    predicted = X.flatten()
    truth = y.flatten()
    
    # our masks will consist of ones and zeros
    # summing the result of their product gives us the cross section
    overlap = np.sum(predicted * truth)
    total_surface_area = np.sum(predicted + truth)
    
    # passing our calculated values to the formula
    return 2 * overlap / total_surface_area


print(f'The dice coefficient for 1 wrongly labeled pixel in a 3*3 image is: {dice_coefficient(X, Y)}')
print('(2 * 2 overlapping "1" pixels / 5 total "1" surface area)')