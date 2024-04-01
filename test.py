import numpy as np

# Create a 3D numpy array with random numbers
array_3d = np.array([0.3,-0.1,0.01])
print(array_3d)
# Take the absolute value of every element
absolute_array = np.abs(array_3d)
print(absolute_array)
# Set elements that are less than 0.1 to 0.1
adjusted_array = np.where(absolute_array < 0.1, 0.1, absolute_array)
print(adjusted_array)