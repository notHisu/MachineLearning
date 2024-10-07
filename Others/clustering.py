import matplotlib.pyplot as plt

X = [(2,3),(3,4),(5,6),(8,9), (9,10)]

x_coords = [point[0] for point in X]
y_coords = [point[1] for point in X]

plt.scatter(x_coords, y_coords, color="blue", s=100, label="Data Points")

plt.title("Scatter Plot of Points")
plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")
plt.grid(True)
plt.legend()

plt.show()