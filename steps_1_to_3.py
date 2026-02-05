#%% [markdown]
# # Step 1: Identify a question that the dataset can answer:
# **College Completion Dataset:** \
# Based on the amount of aid a school provides to students, both merit and financial, is the students at that school are more likely to complete college?

# **Job Placement Dataset:** \
# Is there a correlation between the performance on standardized tests and degree performance percentile and whether you have gotten a job or not? 

# %% [markdown]
# # Step 2: Independent Business Metric and Data Cleaning:
# **College Completion Dataset:** 
# - **Independent Business Metric:** \
# `awards_per_state_value` = The number of awards per 100 full-time undergraduates compared to the state average. \ 

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
# Turning necessary variables into categorical variables:
cat_cols = ['state', 'level', 'control']

for col in cat_cols:
    COLLEGE[col] = COLLEGE[col].astype('category')

COLLEGE.info()

# %%
# Dropping unnecessary columns:
cols_by_name = [
    "site", "long_x", "lat_y",
    "med_sat_percentile", "med_sat_value", "endow_value", "basic",
    "endow_percentile", "unitid", "chronname", "city", "grad_100_value",
    "grad_100_percentile", "grad_150_value", "grad_150_percentile"
]

cols_by_index = COLLEGE.columns[34:63]

cols_to_drop = [
    c for c in cols_by_name + list(cols_by_index)
    if c in COLLEGE.columns
]

if cols_to_drop:
    COLLEGE = COLLEGE.drop(columns=cols_to_drop)

# %%
# Turning Boolean columns into 0s and 1s:
bool_cols = ['hbcu', 'flagship']

for col in bool_cols:
    if col in COLLEGE.columns:
        COLLEGE[col] = (COLLEGE[col] == "X").astype(int)
    else:
        pass 


# %%
# One-Hot Encoding categorical variables:
one_hot_columns = ['level', 'control']

cols_to_encode = [c for c in one_hot_columns if c in COLLEGE.columns]

if cols_to_encode:
    COLLEGE = pd.get_dummies(COLLEGE, columns=cols_to_encode)
else:
    pass 


# %% 
# Standardizing and Scaling numerical columns:
columns = COLLEGE.columns[5:28]
scaler = MinMaxScaler()

for col in columns:
    if col in COLLEGE.columns:
        COLLEGE[[col]] = scaler.fit_transform(COLLEGE[[col]])
    else:
        pass

########################################################################################
########################################################################################
# %%
# Data Preparation and Cleaning for Job Placement Dataset:
JOB = pd.read_csv('Placement_Data_Full_Class.csv')
JOB.info()

# %%
# Turning necessary variables into categorical variables:
cat_cols = ["gender" , "ssc_b", "hsc_b", "hsc_s", "degree_t",
            "specialisation"]

for col in cat_cols:
    JOB[col] = JOB[col].astype('category')

JOB.info()

# %%
# Dropping unnecessary columns:
cols_by_name = ["salary"]

cols_to_drop = [
    c for c in cols_by_name
    if c in JOB.columns
]

if cols_to_drop:
    JOB =JOB.drop(columns=cols_to_drop)


# %%
# One-Hot Encoding categorical variables:
one_hot_columns = ["gender","ssc_b", "hsc_b", "hsc_s", "degree_t", "specialisation",
                   'status', 'workex']

cols_to_encode = [c for c in one_hot_columns if c in JOB.columns]

if cols_to_encode:
    JOB = pd.get_dummies(JOB, columns=cols_to_encode)
else:
    pass 


# %% 
# Standardizing and Scaling numerical columns:
columns = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
scaler = MinMaxScaler()

for col in columns:
    if col in JOB.columns:
        JOB[[col]] = scaler.fit_transform(JOB[[col]])
    else:
        pass
# %%