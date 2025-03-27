import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import krippendorff
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import QUESTIONNAIRES_FILE, EVAL_PLOTS

# Set clean visualization style
sns.set_theme(style="whitegrid")


def normalize(data, lower_b=-1, upper_b=1):
    """Normalize data to a specified range"""
    minimum, maximum = min(data), max(data)
    return lower_b + (upper_b - lower_b) * (data - minimum) / (maximum - minimum)


def find_questionnaire_files(directory='data'):
    """Find all reconstructed questionnaire CSV files"""
    pattern = re.compile(r'questionnaires_reconstructed_([^.]+)\.csv')
    return [(match.group(1), os.path.join(directory, filename))
            for filename in os.listdir(directory)
            if (match := pattern.match(filename))]


def calculate_krippendorff_metrics(original_df, reconstructed_files):
    """Calculate Krippendorff's alpha metrics for each algorithm"""
    results = {}

    for algorithm_name, file_path in reconstructed_files:
        print(f"Processing '{algorithm_name}'...")
        recon_df = pd.read_csv(file_path)

        # Calculate Krippendorff's alpha for each user
        kripp_values = []
        for i, row in original_df.iterrows():
            # Prepare vectors
            original = row.drop('uuid').values
            original = np.where(original > 9, 5, original)  # Handle values > 9
            reconstructed = recon_df.iloc[i].drop('uuid').values

            # Normalize vectors
            original_norm = normalize(original)
            reconstructed_norm = normalize(reconstructed)

            # Calculate Krippendorff's alpha
            reliability_data = np.array([original_norm, reconstructed_norm], dtype=float)
            alpha = krippendorff.alpha(reliability_data, level_of_measurement='interval')
            kripp_values.append(alpha)

        # Store values and calculate key statistics
        results[algorithm_name] = {
            'values': kripp_values,
            'mean': np.mean(kripp_values),
            'median': np.median(kripp_values),
            'std': np.std(kripp_values),
            'quartiles': np.percentile(kripp_values, [25, 50, 75])
        }

        print(f"  Mean α: {results[algorithm_name]['mean']:.3f}, "
              f"Median α: {results[algorithm_name]['median']:.3f}")

    return results


def plot_krippendorff_analysis(results, output_dir=EVAL_PLOTS):
    """Create comprehensive Krippendorff analysis visualizations"""
    # Sort algorithms by mean performance
    algorithms = sorted(results.keys(), key=lambda x: results[x]['mean'])

    # Create a single figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                   gridspec_kw={'width_ratios': [1, 1.5]})

    # 1. Bar chart with mean values
    means = [results[algo]['mean'] for algo in algorithms]
    stds = [results[algo]['std'] for algo in algorithms]
    colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(algorithms)))

    bars = ax1.barh(algorithms, means, alpha=0.8, color=colors)
    # bars = ax1.barh(algorithms, means, xerr=stds, alpha=0.8, color=colors) # also plots line for std

    # Add text labels
    for i, bar in enumerate(bars):
        ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{means[i]:.3f}", va='center')

    ax1.set_xlabel("Mean Krippendorff's Alpha (α)")
    ax1.set_title("Recommender System Performance")
    ax1.set_xlim(0, max(means) * 1.2)

    # 2. Boxplot for distribution
    box_data = [results[algo]['values'] for algo in algorithms]

    box = ax2.boxplot(box_data, vert=False, patch_artist=True,
                      medianprops={'color': 'black', 'linewidth': 1.5})

    # Add individual points with jitter
    for i, data in enumerate(box_data):
        y = np.random.normal(i + 1, 0.05, size=len(data))
        ax2.scatter(data, y, alpha=0.4, s=20, c='navy', edgecolor=None)

    # Color boxes
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_yticklabels(algorithms)
    ax2.set_xlabel("Krippendorff's Alpha (α)")
    ax2.set_title("Distribution Across Users")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'krippendorff_summary.png'))
    plt.close()

    # Create heatmap of top/bottom performers
    create_performance_heatmap(results, algorithms, output_dir)

    # Create full user-algorithm heatmap
    create_full_user_heatmap(results, algorithms, output_dir)


def create_performance_heatmap(results, algorithms, output_dir):
    """Create a heatmap showing best and worst performing users for each algorithm"""
    # Get all user krippendorff values
    all_values = np.array([results[algo]['values'] for algo in algorithms])

    # Identify interesting users (best/worst performers)
    user_means = np.mean(all_values, axis=0)

    # Get indices of 5 best and 5 worst users
    n_selected = min(5, len(user_means))
    best_indices = np.argsort(user_means)[-n_selected:][::-1]
    worst_indices = np.argsort(user_means)[:n_selected]
    selected_indices = np.concatenate([best_indices, worst_indices])

    # Create matrix for heatmap
    matrix = all_values[:, selected_indices].T

    # Create labels
    row_labels = [f"Best {i + 1}" for i in range(n_selected)] + \
                 [f"Worst {i + 1}" for i in range(n_selected)]

    # Create heatmap
    plt.figure(figsize=(10, 4))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # Draw heatmap with annotations
    sns.heatmap(matrix, cmap=cmap, center=0.5, annot=True, fmt=".2f",
                xticklabels=algorithms, yticklabels=row_labels)

    plt.title("Top and Bottom Performing Users Across Recommender Systems")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'krippendorff_users_extremes.png'))
    plt.close()


def create_full_user_heatmap(results, algorithms, output_dir):
    """Create a heatmap comparing all users across algorithms"""
    # Get all user krippendorff values
    all_values = np.array([results[algo]['values'] for algo in algorithms])
    n_algorithms = len(algorithms)
    n_users = len(all_values[0])

    # If too many users, cluster them or use a sampling approach
    max_visible_users = 40  # Maximum number of users to show individually

    if n_users <= max_visible_users:
        # Show all users
        matrix = all_values.T  # Shape: [n_users, n_algorithms]
        user_labels = [f"User {i + 1}" for i in range(n_users)]
    else:
        # Cluster users based on their Krippendorff patterns across algorithms
        from sklearn.cluster import KMeans

        # Determine optimal number of clusters (between 10 and max_visible_users)
        n_clusters = min(max(10, max_visible_users // 2), max_visible_users)

        # Standardize the data for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(all_values.T)

        # Cluster users
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)

        # Create a matrix of cluster centroids
        matrix = np.zeros((n_clusters, n_algorithms))
        for i in range(n_clusters):
            cluster_indices = np.where(clusters == i)[0]
            cluster_size = len(cluster_indices)
            matrix[i] = np.mean(all_values[:, cluster_indices], axis=1)

        # Create labels for clusters indicating size
        user_labels = [f"Cluster {i + 1}\n({np.sum(clusters == i)} users)"
                       for i in range(n_clusters)]

    # Create heatmap
    plt.figure(figsize=(10, max(8, matrix.shape[0] * 0.3)))

    # Choose colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw heatmap
    sns.heatmap(matrix, cmap=cmap, center=0.5,
                annot=True if matrix.shape[0] <= 20 else False,
                fmt=".2f", linewidths=0.5,
                xticklabels=algorithms, yticklabels=user_labels)

    plt.title(f"Krippendorff's Alpha: All Users across Recommender Systems")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'krippendorff_users_heatmap.png'))
    plt.close()


if __name__ == "__main__":
    # Load original data
    print("Loading original questionnaires (ground truth)")
    questionnaires_df = pd.read_csv(QUESTIONNAIRES_FILE)

    # Find reconstructed files
    reconstructed_files = find_questionnaire_files()
    print(f"Found {len(reconstructed_files)} types of reconstructed questionnaires")

    # Calculate metrics
    results = calculate_krippendorff_metrics(questionnaires_df, reconstructed_files)

    # Create visualizations
    print("\nGenerating plots...")
    plot_krippendorff_analysis(results)

    print("Done! All plots saved to the evaluation plots directory.")
