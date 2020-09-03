import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def display_matrix(matrix):
    plt.figure(figsize=(10, 10))
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap='Blues')
    plt.show()


def display_dominance_map(map):
    plt.figure(figsize=(10, 10))
    matrix = np.max(map, axis=0).reshape((21, 21)).round(2)
    reshaped_map = map.reshape((4, 21, 21))
    colors = ["Oranges", "Reds", "Greens", "Purples"]
    for i, color in enumerate(colors):
        sns.heatmap(matrix, mask=(reshaped_map[i, :, :] <= 0), annot=True, cmap=color, cbar=False)
    plt.show()
