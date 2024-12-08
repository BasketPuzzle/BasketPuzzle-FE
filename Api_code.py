#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI, Query
from typing import List, Union
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import itertools

# FastAPI 앱 생성
app = FastAPI()

# 데이터 로드
print("Loading data...")

try:
    df_raw = pd.read_csv("C:/Users/znxls/Downloads/groceries - groceries.csv")
except FileNotFoundError:
    print("File does not exist. Please check the file path.")
    raise

print("Data loading completed.")

# 데이터 전처리
print("Data preprocessing in progress...")
item_cols = [c for c in df_raw.columns if c.lower().startswith('item ') and 'item(s)' not in c.lower()]

if len(item_cols) == 0:
    raise ValueError("Item columns could not be found. Please check the dataset structure.")

transactions = []
for idx, row in df_raw[item_cols].iterrows():
    basket = row.dropna().tolist()
    if len(basket) > 0:
        transactions.append([str(item).strip().lower() for item in basket if str(item).strip() != ""])

if len(transactions) == 0:
    raise ValueError("No valid transactions found. Please check the CSV file or preprocessing logic.")

all_items = [item for basket in transactions for item in basket]
unique_items = set(all_items)

print("Converting to binary matrix using TransactionEncoder...")
te = TransactionEncoder()
te_array = te.fit_transform(transactions, sparse=False)
df = pd.DataFrame(te_array, columns=te.columns_)
print("Binary matrix transformation completed.")

print("Extracting frequent itemsets using Apriori algorithm...")
freq_itemsets = apriori(df, min_support=0.01, use_colnames=True)
print("Frequent itemsets extraction completed.")

# Custom Association Rules 함수
def custom_association_rules(freq_itemsets, metric="confidence", min_threshold=0.3):
    if 'itemsets' not in freq_itemsets.columns or 'support' not in freq_itemsets.columns:
        raise ValueError("'freq_itemsets' DataFrame must have 'itemsets' and 'support' columns.")

    support_dict = {frozenset(item): supp for item, supp in zip(freq_itemsets['itemsets'], freq_itemsets['support'])}
    rules_list = []

    for itemset in freq_itemsets['itemsets']:
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in map(frozenset, itertools.combinations(itemset, r)):
                consequent = itemset - antecedent
                support_itemset = support_dict[itemset]
                support_antecedent = support_dict[antecedent]
                support_consequent = support_dict.get(consequent, None)
                if support_consequent is None:
                    continue
                confidence = support_itemset / support_antecedent
                lift = confidence / support_consequent
                if metric == "confidence":
                    if confidence >= min_threshold:
                        rules_list.append((antecedent, consequent, support_antecedent, support_consequent, support_itemset, confidence, lift))
                elif metric == "lift":
                    if lift >= min_threshold:
                        rules_list.append((antecedent, consequent, support_antecedent, support_consequent, support_itemset, confidence, lift))
                else:
                    raise ValueError("The metric must be 'confidence' or 'lift'.")

    columns = ['antecedents', 'consequents', 'antecedent_support', 'consequent_support', 
               'support', 'confidence', 'lift']
    rules_df = pd.DataFrame(rules_list, columns=columns)
    return rules_df.sort_values('lift', ascending=False).reset_index(drop=True)

# 연관규칙 생성
rules = custom_association_rules(freq_itemsets, metric='confidence', min_threshold=0.1)
print("Association rules generated.")

# API 엔드포인트

@app.get("/")
def read_root():
    return {"message": "Welcome to the Market Basket Analysis API"}

@app.get("/get_rules")
def get_rules():
    return rules.to_dict(orient="records")

@app.get("/get_top_items")
def get_top_items():
    item_counts = pd.Series(all_items).value_counts()
    top10_items = item_counts.head(10).reset_index()
    top10_items.columns = ["Item", "Count"]
    return top10_items.to_dict(orient="records")

@app.get("/get_item_info/{item_name}")
def get_item_info(item_name: str):
    item_name = item_name.lower()
    total_sales = sum([1 for basket in transactions if item_name in basket])
    co_items = [i for basket in transactions if item_name in basket for i in basket if i != item_name]
    co_item_counts = pd.Series(co_items).value_counts().head(3).to_dict()
    return {
        "item": item_name,
        "total_sales": total_sales,
        "frequently_purchased_with": co_item_counts
    }

@app.get("/get_recommendations")
def get_recommendations(items: Union[str, List[str]] = Query(...)):
    if isinstance(items, str):
        user_items = frozenset(items.lower().split(","))
    else:
        user_items = frozenset(map(str.lower, items))
    filtered_rules = rules[rules['antecedents'] == user_items]
    if filtered_rules.empty:
        return {"recommendations": []}
    recommendations = filtered_rules.sort_values('lift', ascending=False).iloc[:3]['consequents']
    return {"recommendations": list(recommendations)}

