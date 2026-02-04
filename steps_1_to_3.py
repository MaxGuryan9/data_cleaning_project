#%% [markdown]
# # Step 1: Identify a question that the dataset can answer:
# **College Completion Dataset:** \
# Based on the amount of aid a school provides to students, both merit and financial, whether the students at that school are more likely to complete college?

# **Job Placement Dataset:** \
# Is there a correlation between the performance on standardized tests and degree performance percentile and whether you have gotten a job or not? 

# %% [markdown]
# # Step 2: Independent Business Metric and Data Cleaning:
# **College Completion Dataset:** 
# - **Independent Business Metric:** \
# `awards_per_value_region` = The number of awards per 100 full-time undergraduates compared to the region average. \ 

# **Job Placement Dataset:** 
# - **Independent Business Metric:** \
# `status` = Whether or not the candidate is employed or not (1 = employed, 0 = unemployed).
# %%
# Package Imports:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
# Data Preparation and Cleaning for College Completion Dataset:
COLLEGE = pd.read_csv('cc_institution_details.csv')
COLLEGE.info()

# %%

# %%
# Turning necessary variables into categorical variables:
cat_cols = ['state', 'level', 'control']

for col in cat_cols:
    COLLEGE[col] = COLLEGE[col].astype('category')

COLLEGE.info()

# Dropping unnecessary columns:
cols_by_name = [
    "site", "long_x", "lat_y",
    "med_sat_percentile", "med_sat_value", "endow_value", "basic"
]

cols_by_index = COLLEGE.columns[34:63]

COLLEGE = COLLEGE.drop(columns=cols_by_name + list(cols_by_index))

# Turning Boolean columns into 0s and 1s:
bool_cols = ['hbcu', 'flagship']

for col in bool_cols:
    COLLEGE[col] = (COLLEGE[col] == "X").astype(int)

# One-Hot Encoding categorical variables:
one_hot_columns = ['level', 'control']
COLLEGE = pd.get_dummies(COLLEGE, columns=one_hot_columns)

# Standardizing and Scaling numerical columns:
columns = COLLEGE.columns[9:28]
for col in columns:
    COLLEGE[[col]] = MinMaxScaler().fit_transform(COLLEGE[[col]])

# %%
