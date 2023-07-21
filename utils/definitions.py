import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
plt.style.use("ggplot")

feature_cols = [f"Feature_{i}" for i in range(1, 26)]

intra_ret_feat_cols = [f"Ret_{i}" for i in range(2, 121)]
intra_ret_target_cols = [f"Ret_{i}" for i in range(121, 181)]
intra_ret_cols = intra_ret_feat_cols + intra_ret_target_cols

inter_ret_feat_cols = ["Ret_MinusTwo", "Ret_MinusOne"]
inter_ret_target_cols = ["Ret_PlusOne", "Ret_PlusTwo"]
inter_ret_cols = inter_ret_feat_cols + inter_ret_target_cols

all_feature_cols = feature_cols + intra_ret_feat_cols + inter_ret_feat_cols
target_cols = intra_ret_target_cols + inter_ret_target_cols

train_df = pd.read_csv("data/raw/train.csv", index_col=0)
test_df = pd.read_csv("data/raw/test_2.csv", index_col=0)

train_groups = train_df.Feature_7
test_groups = test_df.Feature_7

weight_intraday = train_df.Weight_Intraday.values
weight_interday = train_df.Weight_Daily.values