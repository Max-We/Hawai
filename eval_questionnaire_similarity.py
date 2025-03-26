import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
import krippendorff

from config import QUESTIONNAIRES_FILE, QUESTIONNAIRES_RECONSTRUCTED_FILE


def normalize(data, lower_b=-2, upper_b=2):
    minimum, maximum = min(data), max(data)
    return lower_b + (upper_b - lower_b) * (data - minimum) / (maximum - minimum)


if __name__ == "__main__":
    print("Loading questionnaires")
    questionnaires_df = pd.read_csv(QUESTIONNAIRES_FILE)
    print("Loading reconstructed questionnaires")
    questionnaires_r_df = pd.read_csv(QUESTIONNAIRES_RECONSTRUCTED_FILE)

    # Calculate similarity between original and reconstructed questionnaires
    uuids = questionnaires_df['uuid']
    similarities = []
    metrics = {'Pearson': [], 'Spearman': [], 'Krippendorff': []}
    for i, row in questionnaires_df.iterrows():
        # Create numpy arrays from questionnaire values
        user_vector = row.drop('uuid').values
        user_vector_r = questionnaires_r_df.iloc[i].drop('uuid').values

        # Replace values >9 with 5
        user_vector = np.where(user_vector > 9, 5, user_vector)
        # Normalize: center around 0, range [-1, 1]
        user_vector = normalize(user_vector, -1, 1)
        user_vector_r = normalize(user_vector_r, -1, 1)

        # Calculate Pearson correlation
        pearson_corr, _ = pearsonr(user_vector, user_vector_r)
        print(f"UUID <{uuids[i]}>, pearson-corr:\t{round(pearson_corr, 3)}")

        # Calculate Spearman correlation
        spearman_corr, _ = spearmanr(user_vector, user_vector_r)
        print(f"UUID <{uuids[i]}>, spearman-corr:\t{round(spearman_corr, 3)}")

        # Calculate Krippendorff's alpha
        # Reshape data for krippendorff.alpha
        reliability_data_matrix = np.array([user_vector, user_vector_r], dtype=float)
        kripp_alpha = krippendorff.alpha(reliability_data_matrix, level_of_measurement='interval')
        print(f"UUID <{uuids[i]}>, krippendorff-alpha:\t{round(kripp_alpha, 3)}")

        # Only visualize first 5 users
        if len(metrics['Pearson']) < 5:
            metrics['Pearson'].append(pearson_corr)
            metrics['Spearman'].append(spearman_corr)
            metrics['Krippendorff'].append(kripp_alpha)

        print("-" * 50)

    plt.figure(figsize=(10, 6))
    pd.DataFrame(metrics).plot(kind='bar')
    plt.xlabel('User Index')
    plt.ylabel('Score')
    plt.title('Similarity Metrics Comparison')
    plt.tight_layout()
    plt.savefig('similarity_metrics.png')
    plt.show()
