import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv('/Users/elizabethearhart/email_marketing_proj/email-marketing-segmentation/direct_email_marketing_database.csv')

print("Column Names:", data.columns)

data.fillna(method='ffill', inplace=True)

numbers_to_normalize = [
    'upgrade_price', 'age', 'n_purchase', 'discount_purchase', 'n_reward',
    'n_first_class', 'n_second_class', 'n_third_class', 'n_fourth_class',
    'avg_npassengers', 'avg_price', 'sdt_dev_price', 'avg_distance',
    'sdt_dev_distance', 'since_last_purchase', 'n_sent_reminder',
    'n_open_reminder', 'avg_opens_reminder', 'n_click_reminder',
    'avg_clicks_reminder', 'n_sent_upgrade', 'n_open_upgrade',
    'avg_opens_upgrade', 'n_click_upgrade', 'avg_clicks_upgrade',
    'n_sent_discount', 'n_open_discount', 'avg_opens_discount',
    'n_click_discount', 'avg_clicks_discount', 'price', 'days_2_trip', 
    'distance', 'n_sessions', 'n_bounces', 'n_hits',
    'total_session_duration', 'total_revenue', 'conversions', 'n_search',
    'n_path', 'avg_hits', 'avg_session_duration', 'avg_revenue'
]
    
scaler = StandardScaler()
data[numbers_to_normalize]= scaler.fit_transform(data[numbers_to_normalize])
data['is_second_class'] = data['is_second_class'].astype(int)
data['success'] = data['success'].astype(int)

segments_based_on = [
    'age', 'n_purchase', 'discount_purchase', 'n_reward', 
    'n_sent_reminder', 'n_open_reminder', 'n_click_reminder',
    'n_sessions', 'total_revenue', 'conversions'
]

kmeans = KMeans(n_clusters=5, random_state=0)
clusters = kmeans.fit_predict(data[segments_based_on])

data['Cluster'] = clusters

evaluation = silhouette_score(data[segments_based_on], data['Cluster'])
print(f'Silhouette Score: {evaluation}')

cluster_summary = data.groupby('Cluster').mean()
print(cluster_summary)