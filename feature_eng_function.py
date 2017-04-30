import os
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

def feature_eng_forest(data_file_path, soil_file_path):
	try:
		df = pd.read_csv(data_file_path, sep=',', header=0, index_col='Id')
		soil_types = pd.read_csv(soil_file_path).set_index('Soil Type')
	except:
		df = pd.read_csv(filepath, sep=',', header=0)
		soil_types = pd.read_csv(soil_file_path).set_index('Soil Type')
	def labelSoilType(row):
    	for i in range(len(row)):
        	if row[i] == 1:
            	return 'Soil_Type'+str(i)
    df['Soil Type'] = df[['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7',
       'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
       'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15',
       'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
       'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23',
       'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 'Soil_Type27',
       'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
       'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35',
       'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39',
       'Soil_Type40']].apply(lambda row: labelSoilType(row), axis=1)
    df = pd.merge(df, soil_types, how='left', left_on='Soil Type', right_index=True)
    del df['Soil Type']
    print("Dropped the following columns: \n")
    for i in range(df.shape[1]-2, 14, -1):
    	if df[df.columns.tolist()[i]].std() == 0:
        	print df.columns.tolist()[i]
        	df = df.drop(df.columns.tolist()[i], 1)
    df['Direct_Distance_To_Hydrology']=np.sqrt((df.Vertical_Distance_To_Hydrology**2) + (df.Horizontal_Distance_To_Hydrology**2)).astype(float).round(2)
    def azimuth_to_abs(x):
    	if x>180:
        	return 360-x
    	else:
        	return x

    df['Aspect2'] = df.Aspect.map(azimuth_to_abs)
    df['Above_Sealevel'] = (df.Vertical_Distance_To_Hydrology>0).astype(int)
    bins = [0, 2600, 3100, 8000]
	group_names = [1, 2, 3]
	df['Elevation_Bucket'] = pd.cut(df['Elevation'], bins, labels=group_names)
	df['3PM_0_Hillshade'] = (df.Hillshade_3pm==0).astype(int)
	soil_types= df.columns.tolist()[14:53]
	column_list = df.columns.tolist()
	del column_list[14:54]
	column_list.insert(1, 'Elevation_Bucket')
	column_list.insert(3, 'Aspect2')
	column_list.insert(5, 'Direct_Distance_To_Hydrology')
	column_list.insert(8, 'Above_Sealevel')
	column_list.insert(13, '3PM_0_Hillshade')
	column_list=column_list[:-5]
	column_list.extend(soil_types)
	df = df[column_list]
	for i in range(df.shape[1]-2, 92, -1):
    	if df[df.columns.tolist()[i]].std() == 0:
        	df = df.drop(df.columns.tolist()[i], 1)
    for i in range(df.shape[1]-1):
    	for j in range(54):
        	if i != j:
            	df[df.columns.tolist()[i]+"_"+df.columns.tolist()[j]] = df[df.columns.tolist()[i]]*df[df.columns.tolist()[j]]
    for i in range(df.shape[1]-2, 92, -1):
    	if df[df.columns.tolist()[i]].std() == 0:
        	df = df.drop(df.columns.tolist()[i], 1)

    return df



