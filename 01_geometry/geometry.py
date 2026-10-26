import numpy as np

# Three points in 3D
points = np.array([
    [0.0, 0.0, 0.0], #A
    [3.0, 0.0, 0.0], #B
    [3.0, 4.0, 2.0], #C
])


# Distance between any two points

def distance(a, b):
    return np.linalg.norm(a - b)

print("Original distances:")
print("A-B:", distance(points[0], points[1]))
print("B-C:", distance(points[1], points[2]))
print("A-C:", distance(points[0], points[2]))

# Rotate 90 degrees around the Z axis
rotation = np.array([
    [0.0, -1.0, 0.0],
    [1.0,  0.0, 0.0],
    [0.0,  0.0, 1.0],

])

rotated = points @ rotation.T

print("\nRotated coordinates:")
print(rotated)
