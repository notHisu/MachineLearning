import matplotlib.pyplot as plt
import numpy as np

# Check version
# print(plt.__version__)


# Pyplot

# xpoints = np.array([0 ,6])
# ypoints = np.array([0, 250])

# plt.plot(xpoints, ypoints)
# plt.show()

# Plotting Without Line

# xpoints = np.array([1, 8])
# ypoints = np.array([3, 10])

# plt.plot(xpoints, ypoints, 'o') # 'o' is used for plotting without line
# plt.show()

# Multiple Points

# xpoints = np.array([1, 2, 6, 8])
# ypoints = np.array([3, 8, 1, 10])

# plt.plot(xpoints, ypoints)
# plt.show()

# Default X-Points

# ypoints = np.array([3, 8, 1, 10, 5, 7])

# plt.plot(ypoints)
# plt.show()


# Markers 

# Use marker to emphasize the points   

# ypoints = np.array([3, 8, 1, 10, 5, 7])

# plt.plot(ypoints, marker = '*') # Can replace 'o'. Checkout marker reference in https://matplotlib.org/stable/api/markers_api.html
# plt.show()

# Format Strings 
# syntax: marker|line|color

# ypoints = np.array ([3, 8, 1, 10])

# plt.plot(ypoints, 'o:r') # Checkout line reference in https://matplotlib.org/stable/api/markers_api.html, color reference in https://matplotlib.org/stable/gallery/color/named_colors.html
# plt.show()

# Marker Size

# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, marker = 'o', ms = 20) 
# plt.show()

# Marker Color

# mec parameter is for marker edge color

# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r')
# plt.show()

# mfc parameter is for marker face color

# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, marker = 'o', ms = 20, mfc = 'r') 
# plt.show()

# Can use both mec and mfc


# Linestyle

# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, linestyle = 'dotted')
# plt.show()

# Shorter Syntax
# linestyle = ls
# dotted = :
# dashed = --
# Checkout linestyle reference in https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

# Line Color 

# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, color = 'r')
# plt.show()

# Line Width

# ypoints = np.array([3, 8, 1, 10])

# plt.plot(ypoints, linewidth = '20.5')
# plt.show()

# Multiple Lines 

# y1 = np.array([3, 8, 1, 10])
# y2 = np.array([6, 2, 7, 11])

# plt.plot(y1)
# plt.plot(y2)
# plt.show()


# Labels and Title

# Create labels for the x and y axis

# x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
# y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

# plt.plot(x, y)

# plt.xlabel("Average Pulse")
# plt.ylabel("Calorie Burnage")

# plt.show()

# Create title for a plot

# x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
# y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

# plt.plot(x, y)

# plt.title("Sports Watch Data")

# plt.show()

# Set Font Prorerties for Title and Labels

# x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
# y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

# font1 = {'family':'serif','color':'blue','size':20}
# font2 = {'family':'serif','color':'darkred','size':15}

# plt.title("Sports Watch Data", fontdict = font1)
# plt.xlabel("Average Pulse", fontdict = font2)
# plt.ylabel("Calorie Burnage", fontdict = font2)

# plt.plot(x, y)
# plt.show()

# Position the Title

# x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
# y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

# plt.title("Sports Watch Data", loc = 'right')

# plt.plot(x, y)
# plt.show()


# Add Grid Lines 

# x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
# y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

# plt.title("Sports Watch Data")

# plt.xlabel("Average Pulse")
# plt.ylabel("Calorie Burnage")

# plt.plot(x, y)

# plt.grid()    

# plt.show()

# Specific Grid Lines

# plt.grid(axis = 'y') # 'x' or 'y'

# Set Line Properties of Grid

# plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)


# Subplot

# Display Multiple Plots

# x = np.array([0, 1, 2, 3])
# y = np.array([3, 8, 1, 10])

# plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
# plt.plot(x,y)

# x = np.array([0, 1, 2, 3])
# y = np.array([10, 20, 30, 40])

# plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
# plt.plot(x,y)

# plt.show()

# Add Title for each plot

# x = np.array([0, 1, 2, 3])
# y = np.array([3, 8, 1, 10])

# plt.subplot(1, 2, 1)
# plt.plot(x,y)
# plt.title("First Plot")

# x = np.array([0, 1, 2, 3])
# y = np.array([10, 20, 30, 40])

# plt.subplot(1, 2, 2)
# plt.plot(x,y)
# plt.title("Second Plot")

# Super Title

# plt.suptitle("My First Plot")


# Scatter Plot

# x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
# y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

# plt.scatter(x, y)
# plt.show()

# Compare plots

# #day one, the age and speed of 13 cars:
# x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
# y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
# plt.scatter(x, y)

# #day two, the age and speed of 15 cars:
# x = np.array([2,2,8,1,15,8,12,9,7,3,11,4,7,14,12])
# y = np.array([100,105,84,105,90,99,90,95,94,100,79,112,91,80,85])
# plt.scatter(x, y)

# plt.show()

# Color

# x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
# y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
# plt.scatter(x, y, color = 'hotpink')

# Color each dot

# x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
# colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'black', 'white', 'cyan']

# plt.scatter(x, y, c=colors)

# ColorMap

# x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
# y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])
# colors = np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90])

# plt.scatter(x, y, c=colors, cmap='viridis')
# Checkout Available ColorMaps: https://matplotlib.org/stable/tutorials/colors/colormaps.html

# Size 

# x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
# y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

# plt.scatter(x, y, s=100)

# sizes = np.array([20,50,100,200,500,1000,60,90,10,300,600,800,75])

# plt.scatter(x, y, s=sizes)

# Alpha (transparency)

# x = np.array([5,7,8,7,2,17,2,9,4,11,12,9,6])
# y = np.array([99,86,87,88,111,86,103,87,94,78,77,85,86])

# plt.scatter(x, y, s=100, alpha=0.5)

# plt.show()


# Bar 

# x = np.array(["A", "B", "C", "D"])
# y = np.array([3, 8, 1, 10])

# plt.bar(x,y)
# plt.show()

# Horizontal Bar

# x = np.array(["A", "B", "C", "D"])
# y = np.array([3, 8, 1, 10])

# plt.barh(x, y)
# plt.show()

# Bar Color 

# plt.barh(x, y, color = "red")

# Bar Width

# plt.barh(x, y, color = "hotpink", height = 0.3)


# Histogram

# y = np.array([35, 25, 25, 15])
# y = np.random.normal(170, 10, 250)

# plt.hist(y)
# plt.show()


# Pie 

# y = np.array([35, 25, 25, 15])
# plt.pie(y)
# plt.show()

# Labels

# y = np.array([35, 25, 25, 15])
# mylabels = ["Apples", "Bananas", "Cherries", "Dates"]
# plt.pie(y, labels = mylabels)
# plt.show()

# Start Angle

# y = np.array([35, 25, 25, 15])
# mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

# plt.pie(y, labels = mylabels, startangle = 90)

# Explode (make wedges stand out)

# y = np.array([35, 25, 25, 15])
# mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

# plt.pie(y, labels = mylabels, explode = (0, 0.1, 0, 0))

# Shadow

# y = np.array([35, 25, 25, 15])
# mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

# plt.pie(y, labels = mylabels, explode = (0, 0.1, 0, 0), shadow = True)

# Colors

# y = np.array([35, 25, 25, 15])
# mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

# mycolors = ["black", "hotpink", "b", "#4CAF50"]

# plt.pie(y, labels = mylabels, colors = mycolors)

# Legend (list of explaination)

# y = np.array([35, 25, 25, 15])
# mylabels = ["Apples", "Bananas", "Cherries", "Dates"]

# plt.pie(y, labels = mylabels)
# plt.legend()
# plt.legend(title = "Four Fruits:") # Legend title 
# plt.show() 

