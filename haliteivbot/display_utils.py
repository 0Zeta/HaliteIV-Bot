import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def display_matrix(matrix):
    plt.figure(figsize=(10, 10))
    sns.heatmap(pd.DataFrame(matrix), annot=True, cmap='Blues')
    plt.show()
