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
    
    shortest_max = df['shortest_path_complexity'].max()
    simplest_max = df['simplest_path_complexity'].max()

    max_complexity = max(shortest_max,simplest_max)
    print(f"max complexity: {df['shortest_path_complexity'].max()} sum of columns: {df['shortest_path_complexity'].sum()}, mean: {df['shortest_path_complexity'].mean()}, median: {df['shortest_path_complexity'].median()}")
    # now for the shortest path
    print(f"max complexity: {df['simplest_path_complexity'].max()} sum of columns: {df['simplest_path_complexity'].sum()}, mean: {df['simplest_path_complexity'].mean()}, median: {df['simplest_path_complexity'].median()}")
    df['simplest_path_complexity'] = df['simplest_path_complexity'] / max_complexity
    df['shortest_path_complexity'] = df['shortest_path_complexity'] / max_complexity
    print(f"max complexity: {df['shortest_path_complexity'].max()} sum of columns: {df['shortest_path_complexity'].sum()}, mean: {df['shortest_path_complexity'].mean()}, median: {df['shortest_path_complexity'].median()}")
    # now for the shortest path
    print(f"max complexity: {df['simplest_path_complexity'].max()} sum of columns: {df['simplest_path_complexity'].sum()}, mean: {df['simplest_path_complexity'].mean()}, median: {df['simplest_path_complexity'].median()}")
    return df

def label_length_outliers(df):
    Q1 = df[['simplest_path_length', 'shortest_path_length']].quantile(0.25)
    Q3 = df[['simplest_path_length', 'shortest_path_length']].quantile(0.75)
    IQR = Q3 - Q1
    def is_outlier(row):
        return (
            (row['shortest_path_length'] < (Q1['shortest_path_length'] - 1.5 * IQR['shortest_path_length'])) or
            (row['shortest_path_length'] > (Q3['shortest_path_length'] + 1.5 * IQR['shortest_path_length']))
        )
    df['length_outliers'] = df.apply(is_outlier, axis=1)
    return df

def label_gridlike_groups(df):
    median_value = df['environment_orientation_order'].median()
    df['gridlike_median'] = df['environment_orientation_order'].apply(lambda x: 'above_median' if x > median_value else 'below_median')
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
    labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1']
    df['gridlike_group'] = pd.cut(df['environment_orientation_order'], bins=bins, labels=labels, include_lowest=True)
    return df
