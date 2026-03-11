import pandas as pd
df = pd.read_csv('Data_Merge/f1_grand_dataset_full.csv')
features = [
    'sector_straight_ratio_S1', 'sector_straight_ratio_S2', 'sector_straight_ratio_S3',
    'sector_slow_corner_ratio_S1', 'sector_slow_corner_ratio_S2',
    'sector_length_km_S1', 'sector_length_km_S2', 'sector_length_km_S3'
]
print([c for c in features if c in df.columns])