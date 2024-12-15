import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_triple_bar_chart(csv_file1, csv_file2, csv_file3):
    """
    Create a triple bar chart to compare the F1 Score across three methods.

    Parameters:
    csv_file1 (str): Path to the first CSV file.
    csv_file2 (str): Path to the second CSV file.
    csv_file3 (str): Path to the third CSV file.
    """
    # Read the CSV files
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    df3 = pd.read_csv(csv_file3)

    feature_indices1 = df1['Feature Index'].tolist()

    f1_scores1 = df1['F1 Score'].tolist()
    f1_scores2 = df2['F1 Score'].tolist()
    f1_scores3 = df3['F1 Score'].tolist()

    # Create the bar chart
    x = np.arange(len(feature_indices1))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - width, f1_scores1, width, label='5 iter')
    ax.bar(x, f1_scores2, width, label='0 iter')
    ax.bar(x + width, f1_scores3, width, label='OR')

    ax.set_ylabel('F1 Score')
    ax.set_xlabel('Feature Index')
    ax.set_title('F1 Score Comparison Across Three Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_indices1)
    ax.legend()

    plt.tight_layout()
    plt.savefig('results/bar_chart.png')

if __name__ == "__main__":
    create_triple_bar_chart('results/regin-5-neg_test_f1_scores.csv', 
                            'results/regin-no-neg_test_f1_scores.csv', 
                            'results/simple-or_test_f1_scores.csv')