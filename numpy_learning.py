import numpy as np

# Creating Arrays

# 1D Array
arr = np.array([1, 2, 3, 4, 5])

# 2D Array
arr = np.array([1, 2, 3, 4], ndim=2)

# Check the number of dimensions
print(arr.ndim)


# Indexing Arrays

# Print the first item of the array
print(arr[0])

# Print the 50 number from the array
arr = np.array([10, 20, 30, 40, 50])
print(arr[4])

arr = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
print(arr[1, 0])

# Print the last item using negative indexing
arr = np.array([10, 20, 30, 40, 50])
print(arr[-1])


# Slicing Arrays

# From the second (including) to the fifth element (not including)
arr = np.array([10, 15, 20, 25, 30, 35, 40])
print(arr[1:5])

# From the third (including) to the fifth
print(arr[2:4])

# Return every other item from (including) the second to the fifth (not including)
print(arr[1:5:2])

# Return every other item from the entire array
print(arr[::2])


# Data Types

# Print the data type of the array
arr = np.array([1, 2, 3, 4])
print(arr.dtype)

# Converting data type 
arr = np.array([1.1, 2.1, 3.1])

newarr = arr.astype('i')

print(newarr)
print(newarr.dtype)


# Copy and View

# Copy
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
arr[0] = 42
print(arr)
print(x)

# View
arr = np.array([1, 2, 3, 4, 5])
x = arr.view()
arr[0] = 42
print(arr)
print(x)


# Shape of an Array

# Check shape of an array
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(arr.shape)

# Change shape from 1D to 2D
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
newarr = arr.reshape(4, 3)

# Change shape from 2D to 1D
arr = np.array([[1, 2, 3], [4, 5, 6]])
newarr = arr.reshape(-1)


# Iterating Arrays

# Iterate on the elements of the following 1-D array
arr = np.array([1, 2, 3])
for x in arr:
    print(x)

# Iterate on the elements of the following 2-D array
arr = np.array([[1, 2, 3], [4, 5, 6]])
for x in arr:
    print(x)


# Joining Arrays

# Join two arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))

# Join two 2-D arrays along rows (axis=1)
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr1, arr2), axis=1)

# Join two 2-D arrays along columns (axis=0)
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr1, arr2), axis=0)

# Joining arrays using stack functions
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.stack((arr1, arr2), axis=1)

# Stacking along rows
arr = np.hstack((arr1, arr2))

# Stacking along columns
arr = np.vstack((arr1, arr2))

# Stacking along height
arr = np.dstack((arr1, arr2))


# Splitting Arrays

# Split the array in 3 parts
arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)

# Split the 2-D array into three 2-D arrays
arr = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
newarr = np.array_split(arr, 3)

# Split the 2-D array into three 2-D arrays along rows
newarr = np.array_split(arr, 3, axis=1)


# Searching Arrays

# Find the index where the value is 4
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)




