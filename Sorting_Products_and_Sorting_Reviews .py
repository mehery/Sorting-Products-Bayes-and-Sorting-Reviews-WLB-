import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("/kaggle/input/lazada-indonesian-reviews/20191002-reviews.csv")


### Sorting Products


def check_df(dataframe, head=5):
    print("################### Shape ####################")
    print(dataframe.shape)
    print("#################### Info #####################")
    print(dataframe.info())
    print("################### Nunique ###################")
    print(dataframe.nunique())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("################## Quantiles #################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("#################### Head ####################")
    print(dataframe.head(head))

check_df(df)



# Data Preparation

# We are removing columns that are unnecessary and contain missing values.
df = df.drop(columns=["originalRating", "boughtDate", "likeCount", "helpful", "relevanceScore", "clientType", "retrievedDate"])

# To calculate the prior Bayes score, I need to know the distribution of the ratings received by each product
df["rating2"]=df["rating"]
df_bay=df.pivot_table("rating2", "itemId", "rating", aggfunc="count")

df_bay.head()



# We are changing the names of the variables we created.
df_bay.columns=["1_point","2_point","3_point","4_point","5_point"]
df_bay = df_bay.reset_index()

# We are filling in missing values with 0.
df_bay = df_bay.fillna(0)

df_bay.head(10)



# Bayes score calculation function

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


# We are creating the Bayes score under the variable name bar_score.

df_bay["bar_score"] = df_bay.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                        "2_point",
                                                                        "3_point",
                                                                        "4_point",
                                                                        "5_point"]]), axis=1)

# We are listing the top 20 products according to their Bayes score.

df_bay.sort_values(ascending=False, by="bar_score").head(20)



### Sorting Reviews


# We are trying to understand the data.

def check_df(dataframe, head=5):
    print("################### Shape ####################")
    print(dataframe.shape)
    print("#################### Info #####################")
    print(dataframe.info())
    print("################### Nunique ###################")
    print(dataframe.nunique())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("################## Quantiles #################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("#################### Head ####################")
    print(dataframe.head(head))

check_df(df)



# We are removing rating2 from the previous section
df = df.drop(columns="rating2", axis=1)

# upVotes cannot be -1, so we are removing it
df = df[(df["upVotes"] > -1)] 

df.head()



# Wilson lower bound calculation function

def wilson_lower_bound(up, down, confidence=0.95):

    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# We are creating the Wilson lower bound score under the variable name wilson_lower_bound.

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["upVotes"], x["downVotes"]), axis=1)


# We are listing the reviews of the top 20 products according to their WLB score.

df.sort_values(ascending=False, by="wilson_lower_bound").head(20)



