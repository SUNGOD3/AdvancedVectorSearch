import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_benchmark_results(file_path='benchmarks/results.json'):
    """Load benchmark results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_performance_comparison(results):
    """Create a comprehensive performance comparison visualization."""
    plt.figure(figsize=(20, 12))
    plt.suptitle('Search Method Performance Comparison', fontsize=16)

    # Prepare data for plotting
    performance_data = []
    for dim_key, dim_results in results.items():
        if not dim_key.startswith('dim_') and not dim_key.startswith('large_dim_'):
            continue
        
        for dataset_name, dataset_results in dim_results.items():
            for metric, metric_results in dataset_results.items():
                for method, method_results in metric_results.items():
                    performance_data.append({
                        'Dimension': dim_key,
                        'Dataset': dataset_name,
                        'Metric': metric,
                        'Method': method,
                        'Total Time': method_results.get('training_time', 0) + method_results.get('time', 0),
                        'Search Time': method_results.get('time', 0),
                        'Training Time': method_results.get('training_time', 0)
                    })

    df = pd.DataFrame(performance_data)

    # Total Time Comparison Subplot
    plt.subplot(2, 2, 1)
    total_time_plot = sns.barplot(x='Method', y='Total Time', hue='Dimension', data=df)
    plt.title('Total Time (Training + Search)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Time (seconds)')
    
    # Add value labels on bars
    for container in total_time_plot.containers:
        total_time_plot.bar_label(container, fmt='%.4f', label_type='edge')
    
    plt.tight_layout()

    # Search Time Comparison Subplot
    plt.subplot(2, 2, 2)
    search_time_plot = sns.barplot(x='Method', y='Search Time', hue='Dimension', data=df)
    plt.title('Search Time Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Search Time (seconds)')
    
    # Add value labels on bars
    for container in search_time_plot.containers:
        search_time_plot.bar_label(container, fmt='%.4f', label_type='edge')
    
    plt.tight_layout()

    # Training Time Comparison Subplot
    plt.subplot(2, 2, 3)
    training_time_plot = sns.barplot(x='Method', y='Training Time', hue='Dimension', data=df)
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Training Time (seconds)')
    
    # Add value labels on bars
    for container in training_time_plot.containers:
        training_time_plot.bar_label(container, fmt='%.4f', label_type='edge')
    
    plt.tight_layout()

    # Metric-wise Performance Subplot
    plt.subplot(2, 2, 4)
    metric_time_plot = sns.barplot(x='Metric', y='Search Time', hue='Method', data=df)
    plt.title('Search Time by Distance Metric')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Search Time (seconds)')
    
    # Add value labels on bars
    for container in metric_time_plot.containers:
        metric_time_plot.bar_label(container, fmt='%.4f', label_type='edge')
    
    plt.tight_layout()

    plt.savefig('benchmarks/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_performance_table(results):
    """Create a detailed performance comparison table."""
    performance_data = []
    for dim_key, dim_results in results.items():
        if not dim_key.startswith('dim_') and not dim_key.startswith('large_dim_'):
            continue
        
        for dataset_name, dataset_results in dim_results.items():
            for metric, metric_results in dataset_results.items():
                for method, method_results in metric_results.items():
                    performance_data.append({
                        'Dimension': dim_key,
                        'Dataset': dataset_name,
                        'Metric': metric,
                        'Method': method,
                        'Total Time': method_results.get('training_time', 0) + method_results.get('time', 0),
                        'Training Time': method_results.get('training_time', 0),
                        'Search Time': method_results.get('time', 0)
                    })

    df = pd.DataFrame(performance_data)
    
    # Create a pivot table for easy comparison
    pivot_table = df.pivot_table(
        values='Search Time', 
        index=['Dimension', 'Dataset', 'Metric'], 
        columns='Method', 
        aggfunc='first'
    )
    
    # Save to CSV
    pivot_table.to_csv('benchmarks/performance_summary.csv')
    
    # Print to console
    print("\nDetailed Performance Summary:")
    print(pivot_table)

def main():
    results = load_benchmark_results()
    create_performance_comparison(results)
    create_detailed_performance_table(results)
    print("Visualization and summary complete. Check:")
    print("- benchmarks/performance_comparison.png")
    print("- benchmarks/performance_summary.csv")

if __name__ == "__main__":
    main()