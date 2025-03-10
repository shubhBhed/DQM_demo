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


# context = gx.get_context()

# context = context.convert_to_file_context()
# print(context)

# datasource = context.sources.add_pandas(name="my_pandas_datasource")

# dataframe = pd.read_csv("v_dqm_kpi.csv")

# print(dataframe)
# data_asset = datasource.add_dataframe_asset(name=name)
# my_batch_request = data_asset.build_batch_request(dataframe=dataframe)

import great_expectations as gx

# Retrieve your Data Context
context = gx.get_context()
assert type(context).__name__ == "EphemeralDataContext"

# Define the Data Source name
data_source_name = "my_data_source"

# Add the Data Source to the Data Context
data_source = context.data_sources.add_pandas(name=data_source_name)
assert data_source.name == data_source_name