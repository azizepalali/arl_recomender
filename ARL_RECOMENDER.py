############################################
# ASSOCIATION RULE LEARNING
############################################

############## Case ##############
# The data set named Online Retail II is a UK-based online sale.
# store's sales between 01/12/2009 - 09/12/2011.
# The product catalog of this company includes souvenirs.
# promotion can be considered as products.
# There is also information that most of its customers are wholesalers.

#################################

# 1. Data Preprocessing
# 2. ARL Preparing the Data Structure (Invoice-Product Matrix)
# 3. Determination of Association Rules
# 4. Preparing the Script of the Project
# 5. Making Product Suggestions to Users at the Basket Stage

############################################
# Data Preprocessing
############################################
# !pip install mlxtend

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# It makes the result appear in a single line.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


# Let's get the dataset
df = pd.read_excel('hafta_3\online_retail.xlsx', sheet_name="Year 2010-2011")
df.info()
df.head()

# for outlier values
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Regular process for online retail dataset
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)


############################################
# ARL Preparing the Data Structure (Invoice-Product Matrix)
############################################
# For this analysis, we wanted to work only for Germany customers.
df_de = df[df['Country'] == "Germany"]

# We wanted to observe the steps before making the identification process.
df_de.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)
df_de.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]
df_de.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]
df_de.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).applymap(
                                                                            lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

# Function is defined for all the steps.
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0).\
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0).\
            applymap(lambda x: 1 if x > 0 else 0)


de_inv_pro_df = create_invoice_product_df(df_de)
de_inv_pro_df = create_invoice_product_df(df_de, id=True)
de_inv_pro_df.head()


# We will need this step more than once so that function is defined.

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df_de,21987)
check_id(df_de, 23235)
check_id(df_de, 22747)

############################################
# Determination of Association Rules
############################################

frequent_itemsets = apriori(de_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head(50)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.sort_values("support", ascending=False).head()

rules.sort_values("lift", ascending=False).head(500)

############################################
# Making Product Suggestions to Users at the Basket Stage
############################################

def arl_recommender(rules_df, product_id, rec_count=1):

    sorted_rules = rules_df.sort_values("lift", ascending=False)

    recommendation_list = []

    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))

    recommendation_list = list({item for item_list in recommendation_list for item in item_list})

    return recommendation_list[:rec_count]

check_id(df_de, 23235) # ['STORAGE TIN VINTAGE LEAF']
arl_recommender(rules, 23235, 1)
arl_recommender(rules, 23235, 2) # ['SPACEBOY BIRTHDAY CARD']
