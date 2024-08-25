import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torchvision
import torchvision.transforms as transforms


# filename = "data.csv"

# df = pd.read_csv(filename)

filename = "cleaned_data.csv"

df = pd.read_csv(filename)

# print(df)

# import csv

# filename = "data.csv"

# with open(filename, mode="r") as file:
#     csv_reader = csv.DictReader(file)
    
#     data = [row for row in csv_reader]
    
#     for row in data:
#         print(row)

# df.info()

# print(df.head(3))

# print(df.tail(3))

# print(df.iloc[5])

# print(df.isnull().sum())

# print(df.describe())

# print(df.corr())

# plt.figure(dpi=300)
# sns.heatmap(df.corr(), annot=True, fmt=".1f")
# plt.show()

# df = df.dropna()
# df = df[(df >= 0).all(axis=1)]
# df.to_csv("cleaned_data.csv", index=False)

# plt.figure(figsize=(8,6))
# plt.scatter(df["height"], df["weight"], color="blue", marker='o')

# plt.title("Scatter Plot of Height vs. Weight")
# plt.xlabel("Height (m)")
# plt.ylabel("Weight (kg)")

# plt.grid(True)
# plt.show()

# data_to_scale = df[["height", "weight", "bmi"]]

# scaler = StandardScaler()

# scaled_data = scaler.fit_transform(data_to_scale)

# df_scaled = pd.DataFrame(scaled_data, columns=["height_scaled", "weight_scaled", "bmi_scaled"])

# plt.figure(figsize=(8,6))
# plt.scatter(df_scaled["height_scaled"], df_scaled["weight_scaled"], color="green", marker='o')

# plt.title("Scatter Plot of Scaled Height vs. Weight")
# plt.xlabel("Scaled Height (m)")
# plt.ylabel("Scaled Weight (kg)")

# plt.grid()
# plt.show()

# data_to_scale = df[["height", "weight", ]]

# scaler = MinMaxScaler()
# scaler = MinMaxScaler(feature_range=(1, 10))

# scaled_data = scaler.fit_transform(data_to_scale)

# df_scaled = pd.DataFrame(scaled_data, columns=["height_scaled", "weight_scaled"])

# print(df_scaled)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

image, label = mnist_train[0]

image_pil = transforms.ToPILImage()(image*0.5 + 0.5)
image_pil.save("image.png")

image = image * 0.5 + 0.5
image_np = image.numpy().squeeze()

plt.imshow(image_np, cmap="gray")
plt.title(f'Label: {label}')
plt.show()
