import pandas as pd
import numpy as np
import warnings
import os,sys
import logging
#warnings.simplefilter(actions='ignore', category=FutureWarning)
from datetime import datetime, timedelta, date
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import shutil
import great_expectations as gx
from great_expectations.core.batch import BatchRequest


#dqm_kpi_df = pd.read_excel(r"C:\Users\shubham.bhedurkar\Documents\poc\v_dqm_kpi.xlsx", sheet_name="Sheet1")
dqm_kpi_df = pd.read_csv("v_dqm_kpi.csv")
print(dqm_kpi_df)
# querystring2 = f'select product, pharmacy, week_end_date, nvl (startform_cnt, 0) startform_cnt, nvl (enrollment_cnt, 0) enrollment_cnt, nvl (graduation_cnt, 0)
# graduation_cnt, nvl (true_grad_cnt, 0) true_grad_cnt, nvl (discontinuation_cnt,0) discontinuation_cnt, nvl (true_discon_cnt,0) true_discon_cnt,
# nvl (neverstart_cnt, 0) neverstart_cnt from (ngca_db) .CHANNEL_MART.v_dqm_kpi'

# cs. execute (querystring2)
# dqm_kpi_df = cs. fetch_pandas_all()
#df2 = dqm_kpi_df[dqm_kpi_df['WEEK_END_DATE'] >= (dqm_kpi_df['WEEK_END_DATE'].max()-timedelta(weeks=26))].fillna(0)
df2 = dqm_kpi_df.copy()
print(df2)

zscore = lambda x: (x - x.mean())/x.std()

kpi = df2.copy()
kpi.drop ('PHARMACY', axis=1, inplace=True)
kpi = kpi.groupby(['PRODUCT','WEEK_END_DATE']).sum().reset_index().sort_values(['PRODUCT','WEEK_END_DATE'], ascending=[True, False])

context = gx.get_context()
print(type(context))
print(hasattr(context, "sources"))
# suite_name = "quantile_suite"
# context.add_or_update_expectation_suite(expectation_suite_name=suite_name)
#expectation_suite_name = "my_suite"

#IQR upper and lower bounds for KPIs
kpi['PRODUCT']=kpi['PRODUCT'].astype('string')    #quanatile issue 
kpi['STARTFORM_CNT']=kpi['STARTFORM_CNT'].astype('float') 
kpi['ENROLLMENT_CNT']=kpi['ENROLLMENT_CNT'].astype('float')
kpi['GRADUATION_ CNT']=kpi['GRADUATION_CNT'].astype('float') 
kpi['TRUE_GRAD_CNT']=kpi['TRUE_GRAD_CNT'].astype('float')
kpi['DISCONTINUATION_CNT']=kpi ['DISCONTINUATION_CNT'].astype('float')
kpi['TRUE_DISCON_CNT']=kpi['TRUE_DISCON_CNT'].astype('float')
kpi['NEVERSTART_CNT']=kpi['NEVERSTART_CNT'].astype('float')

validator = context.data_sources.pandas_default.read_dataframe(kpi)
print("validator_type:",validator)
#validator = context.sources.load_pandas(kpi)


#validator = context.get_validator(batch_data=kpi,data_asset_name="mydataframe")

#convert pandas dataframe to great Expectations PandasDataset
#df_ge = gx.from_pandas(kpi)
#df_ge = gx.dataset.PandasDataset(df)

#df_ge = gx.dataset.PandasDataset(kpi)
expectation = gx.expectations.ExpectColumnValuesToBeBetween(
    column="PRODUCT", quantiles=[0.25]
)

batch = batch_definition.get_batch(batch_parameters=batch_parameters)

# Test the Expectation
validation_results = batch.validate(expectation)
print(validation_results)

q1 = validator.get_column_quantiles("PRODUCT", quantiles=[0.25])
q3 = validator.get_column_quantiles("PRODUCT", quantiles=[0.75])

print(q1)
print(q2)

#q1 = kpi.groupby ('PRODUCT').quantile(.25)
#q3 = kpi.groupby('PRODUCT').quantile(.75)
# IQR = q3.sub(q1, fill_value=0)
# lower_bound = q1 - (IQR * 1.5)
# upper_bound = 93 + (IQR * 1.5)
# lower_bound = lower_bound.reset_index ()
# upper_bound = upper_bound.reset_index ()
# lower_bound.columns = [str(col) + '_lower' if col not in ('PRODUCT') else 'PRODUCT' for col in lower_bound.columns]
# upper_bound.columns = [str(col) + '_upper' if col not in ('PRODUCT') else 'PRODUCT' for col in upper_bound.columns]

# #outliers based on IQR
# outliers = kpi.merge(lower_bound, on='PRODUCT', how='outer').merge(upper_bound, on='PRODUCT', how='outer')
# cols = list(kpi.columns)

# cols = [e for e in cols if e not in ('PRODUCT', 'WEEK_END_DATE' ) ]
# for col in cols:
#     col_outlier = col + '_outlier'
#     col_lower = col + '_lower'
#     col_upper = col + '_upper'
#     conditions = [
#         (outliers[col] < outliers[col_lower]), 
#         (outliers[col] > outliers[col_upper])]
# choices = ['Low','High']
# outliers[col_outlier] = np.select(conditions, choices, default='Normal')

# #Apply z score to columns in KPI
# cols = list(kpi.columns)
# cols = [e for e in cols if e not in ('PRODUCT','WEEK_END_DATE')]
# for col in cols:
#     zscore = col + '_z_score'
#     kpi[col_zscore] = kpi.groupby('PRODUCT') [col].apply(zscore)

# #merge outliers to kpi
# merge_cols = ['PRODUCT', 'WEEK _END_DATE', 'STARTFORM_CNT_outlier', 'ENROLLMENT_CNT_outliers', 'GRADUATION_CNT_outliers',
#               'TRUE_GRAD_CNT_outlier','DISCONTINUATION_CNT_outlier','TRUE_DISCON_CNT_outlier', 'NEVERSTART_CNT_outlier']

# #previous 4 weeks from KPI table
# kpi = kpi[kpi['WEEK_END_DATE'] >= (kpi['WEEK_END_DATE' ].max()-timedelta(weeks=4))]
    
# #Z score and outlier columns for the maxium week
# kpi_z = kpi[kpi['WEEK_END_DATE'] == kpi['WEEK_END_DATE'].max()]
# kpi_z = kpi_z[['PRODUCT', 'STARTFORM_CNT_z_score', 'ENROLIMENT_CNT_z_score', 'GRADUATION_CNT_z_score', 'TRUE_GRAD_CNT_z_score', 'DISCONTINUATION_CNT_z_score', 'TRUE_DISCON_CNT_z_score', 'NEVERSTART_CNT_z_score', 'STARTFORM_CNT_outlier', 'ENROLIMENT_CNT_outlier', 'GRADUATION_CNT_outlier', 'TRUE_GRAD_CNT_outlier', 'DISCONTINUATION_CNT_outlier', 'TRUE_DISCON_CNT_outlier', 'NEVERSTART_CNT_outlier']]

# #Z score and outlier columns for OCR, LEM, TEC, and PLG 2 weeks prior to the max
# kpi_2weeks_lag = kpi[kpi['WEEK_END_DATE'] == (kpi['WEEK_END_DATE'].max()-timedelta(weeks=2))]
# kpi_2weeks_lag = kpi_2weeks_lag[kpi_2weeks_lag['PRODUCT'].isin(['OCREVUS','LEMTRADA','TECFIDERA','PLEGRIDY'])]
# kpi_2weeks_lag = kpi_2weeks_lag[['PRODUCT', 'STARTFORM_CNT_z_score', 'ENROLIMENT_CNT_z_score', 'GRADUATION_CNT_z_score', 'TRUE_GRAD_CNT_z_score', 'DISCONTINUATION_CNT_z_score', 'TRUE_DISCON_CNT_z_score', 'NEVERSTART_CNT_z_score', 'STARTFORM_CNT_outlier', 'ENROLIMENT_CNT_outlier', 'GRADUATION_CNT_outlier', 'TRUE_GRAD_CNT_outlier', 'DISCONTINUATION_CNT_outlier', 'TRUE_DISCON_CNT_outlier', 'NEVERSTART_CNT_outlier']]

# #merges the maxium week z scores with the 2 week lag to create one column that has the appropriate score for each metric
# merged = pd.merge(kpi_z, kpi_2weeks_lag, on='PRODUCT', how='outer')

# merged ['ENROLIMENT_CNT_z_score'] = np.where(merged['PRODUCT'].isin (['OCREVUS','LEMTRADA']), merged ['ENROLLMENT_CNT_z_score_y'], merged['ENROLLMENT_CNT_z_score_x'])
# merged ['GRADUATION_CNT_z_score'] = np.where(merged['PRODUCT'].isin (['OCREVUS','LEMTRADA']), merged ['GRADUATION_CNT_z_score_y'], merged['GRADUATION_CNT_z_score_x'])
# merged ['TRUE_GRAD_CNT_z_score'] = np.where(merged['PRODUCT'].isin (['OCREVUS']), merged ['TRUE_GRAD_CNT_z_score_y'], merged['TRUE_GRAD_CNT_z_score_x'])
# merged ['TRUE_DISCON_z_score'] = np.where(merged['PRODUCT'].isin (['TECFIDERA', 'PLEGRIDY', 'OCREVUS']), merged ['TRUE_DISCON_z_score_y'], merged['TRUE_DISCON_z_score_x'])
    
# merged ['ENROLIMENT_CNT_outlier'] = np.where(merged['PRODUCT'].isin (['OCREVUS','LEMTRADA']), merged ['ENROLIMENT_CNT_outlier_y'], merged['ENROLIMENT_CNT_outlier_x'])
# merged ['GRADUATION_CNT_outlier'] = np.where(merged['PRODUCT'].isin (['OCREVUS','LEMTRADA']), merged ['GRADUATION_CNT_outlier_y'], merged['GRADUATION_CNT_outlier_x'])
# merged ['TRUE_GRAD_CNT_outlier'] = np.where(merged['PRODUCT'].isin (['OCREVUS']), merged ['TRUE_GRAD_CNT_outlier_y'], merged['TRUE_GRAD_CNT_outlier_x'])
# merged ['TRUE_DISCON_outlier'] = np.where(merged['PRODUCT'].isin (['TECFIDERA', 'PLEGRIDY', 'OCREVUS']), merged ['TRUE_DISCON_outlie_y'], merged['TRUE_DISCON_outlier_x'])

# merged = merged[['PRODUCT', 'ENROLLMENT_CNT_Z_score', 'GRADUATION_CNT_z_score', 'TRUE_GRAD_CNT_z_score', 'TRUE_DISCON_CNT_z_score', 'ENROLLMENT_CNT_outlier','GRADUATION_CNT_outlier', 'TRUE_GRAD_CNT_outlier', 'TRUE_DISCON_CNT_outlier']]

# #Z score and outlier columns for the maxium week
# kpi_max = kpi[kpi['WEEK_END_DATE'] == kpi['WEEK_END_DATE'].max()]
# kpi_max = kpi_max[['PRODUCT', 'STARTFORM_CNT_z_score', 'DISCONTINUATION_CNT_z_score', 'NEVERSTART_CNT_z_score', 'STARTFORM_CNT_outlier', 'DISCONTINUATION_CNT_outlier', 'NEVERSTART_CNT_outlier']]
    
# kpi = kpi.pivot_table(index='PRODUCT', columns='WEEK_END_DATE', values=['STARTFORM_CNT', 'ENROLIMENT _CNT', 'GRADUATION _CNT', 'TRUE _GRAD_CNT', 'DISCONTINUATION _CNT', 'TRUE_DISCON_CNT', 'NEVERSTART_CNT']).reset_index ()
    
# result = kpi.merge(kpi_max, on='PRODUCT', how='inner').merge(merged, on=' PRODUCT', how=' inner')
# result = result.drop( ('PRODUCT',''),1)
# x = result.columns.tolist()
# x = [x[0]]+[x[-14]]+[x[-11]]+[x[25]]+[x[24]]+[x[23]]+[x[22]]+[x[21]]+[x[-8]]+[x[-4]]+[x[10]]+[x[9]]+[x[8]]+[x[7]]+[x[6]]+[x[-7]]+[x[-3]]+[x[15]]+[x[14]]+[x[13]]+[x[12]]+[x[11]]+[x[-6]]+[x[-2]]+[x[35]]+[x[34]]+[x[33]]+[x[32]]+[x[31]]+[x[-13]]+[х[-10]]+[х[5]]+[х[4]]+[x[3]]+[x[2]]+[x[1]]+[x[-5]]+[x[-1]]+[x[30]]+[x[29]]+[x[28]]+[x[27]]+[x[26]]+[x[-12]]+[x[-9]]+[x[20]]+[x[19]]+[x[18]]+[x[17]]+[x[16]]
# result = result [x]
    
# #Excel export result. to_excel (writer, sheet_name='KPI')
# result.to_excel(writer, sheet_name='KPI')
