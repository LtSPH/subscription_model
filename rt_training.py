import pandas as pd
import numpy as np

path = r'C:\Users\sult\Documents\sub_model\training data\real_time.csv'
#print(all_files)
rt_df = pd.read_csv(path)


valid_chgrp = ['Organic Search', 'Social', 'Direct']
rt_df['channelGrouping'] = np.where(rt_df['channelGrouping'].isin(valid_chgrp), rt_df['channelGrouping'], 'Other')

valid_country = ['Singapore', 'Malaysia', 'United States', 'Australia', 'Indonesia']
rt_df['country'] = np.where(rt_df['country'].isin(valid_country), rt_df['country'], 'Other')
rt_df['TOP_Country_8W'] = np.where(rt_df['TOP_Country_8W'].isin(valid_country), rt_df['TOP_Country_8W'], 'Other')

valid_browser = ['Chrome', 'Safari', 'Android Webview', 'Safari (in-app)', 'Firefox', 'Edge']
rt_df['browser'] = np.where(rt_df['browser'].isin(valid_browser), rt_df['browser'], 'Other')

rt_df['content_category'] = np.where(rt_df['content_category'] != 'free content', rt_df['content_category'], 'free')

valid_topsec_8w = ['Newssingapore', 'Newslifestyle', 'Newsasia', 'Newsbusiness', 'Newsworld']
rt_df['CAT_top_Section_8W'] = np.where(rt_df['CAT_top_Section_8W'].isin(valid_topsec_8w), 
                                       rt_df['CAT_top_Section_8W'], 'Other')
rt_df['CAT_top_Section_1234W'] = np.where(rt_df['CAT_top_Section_1234W'].isin(valid_topsec_8w), 
                                       rt_df['CAT_top_Section_1234W'], 'Other')
                                       

rt_df.columns = ['subscribe_flag', 'client_id', 'channel_grouping', 'country',
       'device', 'browser', 'content_category', 'days_since_pub',
       'grid_pv', 'nth_pv', 'article_pv', 'otherpage_pv', 'clicks_on_index',
       'lastweek_count', 'lastweek_minutes', 'Total_PV_8W', 'Num_days_8W','PV_per_active_day_8W', 'CAT_top_Section_8W', 'CAT_Count_sections_8W','CAT_Count_Sections_index_8W', 'CAT_PV_top_Section_8W','CAT_Numdays_top_Section_8W', 'CAT_TopSectionPV_inTotal_8W','FLAG_ACTIVE_1234W', 'PV_1234W', 'Num_days_1234W','PV_per_active_day_1234W', 'CAT_top_Section_1234W','CAT_Count_sections_1234W', 'CAT_Count_sections_index_1234W','CAT_PV_top_Section_1234W', 'CAT_Numdays_top_Section_1234W','CAT_TopSectionPV_inTotal_1234W', 'FLAG_ACTIVE_5678W', 'PV_5678W','Num_days_5678W', 'PV_per_active_day_5678W', 'CAT_top_Section_5678W','CAT_Count_sections_5678W', 'CAT_Count_sections_index_5678W','CAT_PV_top_Section_5678W', 'CAT_Numdays_top_Section_5678W','CAT_TopSectionPV_inTotal_5678W', 'FLAG_ACTIVE_1W', 'PV_1W','Num_days_1W', 'PV_per_active_day_1W', 'CAT_top_Section_1W','CAT_Count_sections_1W', 'CAT_Count_sections_index_1W','CAT_PV_top_Section_1W', 'CAT_Numdays_top_Section_1W','CAT_TopSectionPV_inTotal_1W', 'FLAG_ACTIVE_1D', 'PV_1D', 'Num_days_1D','CAT_top_Section_1D', 'CAT_Count_sections_1D','CAT_Count_sections_index_1D', 'CAT_PV_top_Section_1D','CAT_Numdays_top_Section_1D', 'CAT_TopSectionPV_inTotal_1D','Days_From_Last_Visit_week1_8', 'Visit_interval_avg_week1_8','TOP_Country_8W', 'TOP_Country_PV_8W', 'TOP_Country_days_8W','Num_country_8W', 'Num_country_index_8W']
      
rt_df.columns = [x.lower() for x in rt_df.columns]

numeric_features = list(rt_df.select_dtypes(include=[np.number]).drop(['subscribe_flag', 'clicks_on_index'], axis=1).columns.values)
categorical_features = list(rt_df.select_dtypes(include=['object']).drop(['client_id'], axis=1).columns.values)


from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper

cat_imp = SimpleImputer(strategy='constant', fill_value='missing')
num_imp = SimpleImputer(strategy='constant', fill_value=0)

mapper = DataFrameMapper([
    ([categorical_column], [cat_imp, LabelBinarizer()]) for categorical_column in categorical_features
] + [(numeric_features, num_imp)])

mp = mapper.fit_transform(rt_df.copy())
mp.shape

import pickle

with open('C:/Users/sult/Documents/sub_model/rt_mapper_py3_v2.map', 'wb') as f:
       pickle.dump(mapper, f)


from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
gnb_model = classifier.fit(mp, rt_df['subscribe_flag'])

with open('C:/Users/sult/Documents/sub_model/py3_model_v2.sav', 'wb') as f:
       pickle.dump(gnb_model, f)

