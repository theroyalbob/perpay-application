import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Constants
DATA_DIR = "data"
OUTPUT_DIR = "output"

def load_expansion_stations():
    """Load the list of stations recommended for expansion."""
    expansion_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'expansion_recommendations.csv'))
    return set(expansion_df['station_id'].astype(str))

def process_quarterly_data(year, quarter):
    """Process a single quarter of trip data."""
    file_name = f"indego-trips-{year}-q{quarter}.csv"
    file_path = os.path.join(DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_name} not found")
        return None
    
    # Read the CSV file with low_memory=False to avoid DtypeWarning
    df = pd.read_csv(file_path, low_memory=False)
    
    # Convert timestamps
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['start_date'] = df['start_time'].dt.date
    
    # Calculate daily trips per bike for each station
    daily_stats = df.groupby(['start_station', 'start_date', 'bike_id']).size().reset_index(name='trips')
    daily_stats = daily_stats.groupby(['start_station', 'start_date']).agg({
        'bike_id': 'nunique',
        'trips': 'sum'
    }).reset_index()
    
    daily_stats['trips_per_bike'] = daily_stats['trips'] / daily_stats['bike_id']
    
    return daily_stats

def analyze_daily_anomalies():
    """Analyze daily anomalies for recommended vs non-recommended stations."""
    # Load expansion stations
    expansion_stations = load_expansion_stations()
    
    # Process 2024 data
    all_daily_stats = []
    for quarter in range(1, 5):
        stats = process_quarterly_data(2024, quarter)
        if stats is not None:
            all_daily_stats.append(stats)
    
    if not all_daily_stats:
        print("No data found for 2024")
        return
    
    # Combine all quarters
    daily_stats = pd.concat(all_daily_stats, ignore_index=True)
    
    # Convert station_id to string for consistent comparison
    daily_stats['start_station'] = daily_stats['start_station'].astype(str)
    
    # Calculate z-scores for each day across all stations
    station_scores = []
    
    # Group by date to calculate daily statistics
    for date, day_data in daily_stats.groupby('start_date'):
        # Calculate mean and std across all stations for this day
        day_mean = day_data['trips_per_bike'].mean()
        day_std = day_data['trips_per_bike'].std()
        
        if day_std < 0.0001:  # Skip days with no variation
            continue
            
        # Calculate z-scores for each station on this day
        for _, row in day_data.iterrows():
            station_id = row['start_station']
            z_score = (row['trips_per_bike'] - day_mean) / day_std
            
            # Add to the appropriate station's records
            found = False
            for score in station_scores:
                if score['station_id'] == station_id:
                    score['z_scores'].append(z_score)
                    score['trips_per_bike'].append(row['trips_per_bike'])
                    score['dates'].append(date)
                    found = True
                    break
            
            if not found:
                station_scores.append({
                    'station_id': station_id,
                    'z_scores': [z_score],
                    'trips_per_bike': [row['trips_per_bike']],
                    'dates': [date],
                    'station_type': 'Expansion Recommended' if station_id in expansion_stations else 'Non-Expansion'
                })
    
    # Process accumulated z-scores for each station
    processed_scores = []
    daily_type_scores = {}  # Track daily scores by station type
    
    for station in station_scores:
        if len(station['z_scores']) < 7:  # Skip stations with less than 7 days of data
            continue
            
        station_type = station['station_type']
        
        # Initialize daily tracking for this station type if not exists
        if station_type not in daily_type_scores:
            daily_type_scores[station_type] = {}
        
        # Add each day's score to the appropriate station type
        for date, z_score in zip(station['dates'], station['z_scores']):
            if date not in daily_type_scores[station_type]:
                daily_type_scores[station_type][date] = []
            daily_type_scores[station_type][date].append(z_score)
    
    # Calculate daily means for each station type
    type_daily_means = {}
    for station_type, daily_scores in daily_type_scores.items():
        type_daily_means[station_type] = {
            date: np.mean(scores) for date, scores in daily_scores.items()
        }
    
    # Calculate statistics for each station type using daily means
    type_stats = []
    for station_type, daily_means in type_daily_means.items():
        scores = list(daily_means.values())
        positive_scores = [z for z in scores if z > 0]
        negative_scores = [z for z in scores if z < 0]
        
        # Count the number of stations in this group
        num_stations = len([s for s in station_scores if s['station_type'] == station_type])
        
        # Sum all scores (not average) and normalize by the number of stations
        type_stats.append({
            'station_type': station_type,
            'total_stations': num_stations,
            'avg_score': np.sum(scores) / num_stations,  # Sum and normalize by number of stations
            'score_std': np.std(scores) / num_stations,
            'avg_positive': np.sum(positive_scores) / num_stations if positive_scores else 0,
            'positive_std': np.std(positive_scores) / num_stations if len(positive_scores) > 1 else 0,
            'avg_negative': np.sum(negative_scores) / num_stations if negative_scores else 0,
            'negative_std': np.std(negative_scores) / num_stations if len(negative_scores) > 1 else 0,
            'avg_days_above': len(positive_scores),
            'avg_days_below': len(negative_scores),
            'avg_days_at_mean': len(scores) - len(positive_scores) - len(negative_scores),
            'avg_trips': np.mean([np.mean(s['trips_per_bike']) for s in station_scores if s['station_type'] == station_type]),
            'avg_std': np.mean([np.std(s['trips_per_bike']) for s in station_scores if s['station_type'] == station_type])
        })
    
    type_stats = pd.DataFrame(type_stats)
    
    # Create a DataFrame for individual station scores
    scores_df = []
    for station in station_scores:
        if len(station['z_scores']) < 7:  # Skip stations with less than 7 days of data
            continue
            
        # Sum z-scores (no division by 3)
        normalized_score = sum(station['z_scores'])
        
        scores_df.append({
            'station_id': station['station_id'],
            'normalized_score': normalized_score,
            'station_type': station['station_type']
        })
    
    scores_df = pd.DataFrame(scores_df)
    
    # Create visualization
    plt.figure(figsize=(14, 8))
    
    # Set bar positions
    x = np.arange(len(type_stats))
    width = 0.25  # Reduced width to accommodate three bars
    
    # Plot three separate bars for each station type
    plt.bar(x - width, type_stats['avg_score'], width, label='Total Score',
            yerr=type_stats['score_std'], capsize=5)
    plt.bar(x, type_stats['avg_positive'], width, label='Positive Scores',
            yerr=type_stats['positive_std'], capsize=5)
    plt.bar(x + width, type_stats['avg_negative'], width, label='Negative Scores',
            yerr=type_stats['negative_std'], capsize=5)
    
    plt.title('Daily Utilization Variation by Station Type (2024)', fontsize=16, pad=20)
    plt.ylabel('Normalized Score (sum of daily z-scores)', fontsize=12)
    plt.xticks(x, type_stats['station_type'], rotation=45)
    plt.legend(fontsize=12)
    
    # Add value labels with improved positioning
    for i, row in type_stats.iterrows():
        # Find the maximum value to position labels above
        max_value = max(row['avg_score'], row['avg_positive'], row['avg_negative'])
        
        # Label for total score
        plt.text(i - width, row['avg_score'] + 0.2,
                f'{row["avg_score"]:.1f}',
                ha='center', va='bottom', fontsize=10)
        
        # Label for positive scores
        plt.text(i, row['avg_positive'] + 0.2,
                f'+{row["avg_positive"]:.1f}',
                ha='center', va='bottom', fontsize=10)
        
        # Label for negative scores
        plt.text(i + width, row['avg_negative'] + 0.2,
                f'{row["avg_negative"]:.1f}',
                ha='center', va='bottom', fontsize=10)
        
        # Total stations and days - position below the lowest bar
        min_value = min(row['avg_score'], row['avg_positive'], row['avg_negative'])
        plt.text(i, min_value - 1.0,
                f'n={row["total_stations"]} stations\n'
                f'↑{row["avg_days_above"]:.0f} ↓{row["avg_days_below"]:.0f} ={row["avg_days_at_mean"]:.0f} days\n'
                f'avg {row["avg_trips"]:.1f} trips/bike (±{row["avg_std"]:.2f})',
                ha='center', fontsize=10)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'utilization_variation.png'), dpi=300)
    plt.close()
    
    # Create a second visualization showing the distribution of total scores
    plt.figure(figsize=(14, 8))
    
    # Create box plots for each station type
    data = [scores_df[scores_df['station_type'] == 'Expansion Recommended']['normalized_score'],
            scores_df[scores_df['station_type'] == 'Non-Expansion']['normalized_score']]
    
    plt.boxplot(data, tick_labels=['Expansion Recommended', 'Non-Expansion'])
    
    plt.title('Distribution of Total Utilization Scores by Station Type (2024)', fontsize=16, pad=20)
    plt.ylabel('Normalized Score (sum of daily z-scores)', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'utilization_scores_distribution.png'), dpi=300)
    plt.close()
    
    # Save detailed results
    type_stats.to_csv(os.path.join(OUTPUT_DIR, 'utilization_stats.csv'), index=False)
    scores_df.to_csv(os.path.join(OUTPUT_DIR, 'station_utilization_scores.csv'), index=False)
    
    print("Analysis complete. Results saved to output directory.")

if __name__ == "__main__":
    analyze_daily_anomalies() 