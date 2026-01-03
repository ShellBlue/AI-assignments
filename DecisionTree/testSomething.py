import pandas as pd

dataset = pd.read_csv("DecisionTree/dataset/Dataset_30_records.csv")

n = len(dataset)

split_index = int(n * 0.8)

train_df = dataset.iloc[:split_index]
test_df = dataset.iloc[split_index:]

print(train_df)
print(test_df)