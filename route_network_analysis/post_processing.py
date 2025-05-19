import pandas as pd


def get_turn_count(turn_list):
    turns = 0
    for turn in turn_list:
        #print(turn)
        if "turn" in turn:
            #print("turn found")
            turns += 1
    return turns



def normalize_complexity(df):
    
    shortest_max = df['shortest_complexity'].max()
    simplest_max = df['simplest_complexity'].max()

    max_complexity = max(shortest_max,simplest_max)
    print(f"max complexity: {df['shortest_complexity'].max()} sum of columns: {df['shortest_complexity'].sum()}, mean: {df['shortest_complexity'].mean()}, median: {df['shortest_complexity'].median()}")
    # now for the shortest path
    print(f"max complexity: {df['simplest_complexity'].max()} sum of columns: {df['simplest_complexity'].sum()}, mean: {df['simplest_complexity'].mean()}, median: {df['simplest_complexity'].median()}")
    df['simplest_complexity_norm'] = df['simplest_complexity'] / max_complexity
    df['shortest_complexity_norm'] = df['shortest_complexity'] / max_complexity
    print(f"max complexity: {df['shortest_complexity'].max()} sum of columns: {df['shortest_complexity'].sum()}, mean: {df['shortest_complexity'].mean()}, median: {df['shortest_complexity'].median()}")
    # now for the shortest path
    print(f"max complexity: {df['simplest_complexity'].max()} sum of columns: {df['simplest_complexity'].sum()}, mean: {df['simplest_complexity'].mean()}, median: {df['simplest_complexity'].median()}")
    return df

def label_length_outliers(df):
    # Calculate Q1 and Q3 for each column separately
    Q1_shortest = df['shortest_length'].quantile(0.25)
    Q3_shortest = df['shortest_length'].quantile(0.75)
    IQR_shortest = Q3_shortest - Q1_shortest

    Q1_simplest = df['simplest_length'].quantile(0.25)
    Q3_simplest = df['simplest_length'].quantile(0.75)
    IQR_simplest = Q3_simplest - Q1_simplest

    # Define outlier detection for shortest_length
    def is_shortest_outlier(row):
        return (
            (row['shortest_length'] < (Q1_shortest - 1.5 * IQR_shortest)) or
            (row['shortest_length'] > (Q3_shortest + 1.5 * IQR_shortest))
        )

    # Define outlier detection for simplest_length
    def is_simplest_outlier(row):
        return (
            (row['simplest_length'] < (Q1_simplest - 1.5 * IQR_simplest)) or
            (row['simplest_length'] > (Q3_simplest + 1.5 * IQR_simplest))
        )

    # Apply the outlier detection to each row and create new columns
    df['shortest_length_outlier'] = df.apply(is_shortest_outlier, axis=1)
    df['simplest_length_outlier'] = df.apply(is_simplest_outlier, axis=1)

    return df

def label_gridlike_groups(df):
    median_value = df['environment_orientation_order'].median()
    df['gridlike_median'] = df['environment_orientation_order'].apply(lambda x: 'above_median' if x > median_value else 'below_median')
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
    df['gridlike_group'] = pd.cut(df['environment_orientation_order'], bins=bins, labels=labels, include_lowest=True)
    return df
