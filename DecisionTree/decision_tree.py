import pandas as pd
from graphviz import Digraph

dataset = pd.read_csv("DecisionTree/dataset/Dataset_30_records.csv")
dataset1 = pd.read_csv("DecisionTree/dataset/Dataset__1000_samples_1.csv")
dataset2 = pd.read_csv("DecisionTree/dataset/Dataset__1000_samples_2.csv")
dataset3 = pd.read_csv("DecisionTree/dataset/Dataset__1000_samples_3.csv")

features = [c for c in dataset.columns if c != "Buys_Product"]
features1 = [c for c in dataset1.columns if c != "Buys_Product"]
features2 = [c for c in dataset2.columns if c != "Buys_Product"]
features3 = [c for c in dataset3.columns if c != "Buys_Product"]

def trainTestSplit(dataset, split_val=0.8):

    n = len(dataset)
    split_index = int(n * split_val)

    train_df = dataset.iloc[:split_index].reset_index(drop=True)
    test_df = dataset.iloc[split_index:].reset_index(drop=True)

    return train_df, test_df


dataset, test = trainTestSplit(dataset)
dataset1, test1 = trainTestSplit(dataset1)
dataset2, test2 = trainTestSplit(dataset2)
dataset3, test3 = trainTestSplit(dataset3)


def giniFromCounts(yes, no):
    total = yes + no
    if total == 0:
        return 0.0
    p_yes = yes/total
    p_no = no/total
    return 1 - p_yes**2 - p_no**2

def nodeGini(df):
    counts = df["Buys_Product"].value_counts()
    yes = int(counts.get("Yes", 0))
    no  = int(counts.get("No", 0))
    return giniFromCounts(yes, no)


def computeBestSplit(df, features):
    pair_counts = {}

    for col in features:
        for f_val, y_val in zip(df[col], df["Buys_Product"]):
            key = (col, f_val, y_val)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    agg = {}
    for (feat, fval, yval), count in pair_counts.items():
        if feat not in agg:
            agg[feat] = {}
        if fval not in agg[feat]:
            agg[feat][fval] = {"Yes": 0, "No": 0}
        agg[feat][fval][yval] += count

    gini_split = {}
    for feat, dict_val in agg.items():
        total_n = sum(c["Yes"] + c["No"] for c in dict_val.values())

        weighted_gini = 0.0
        for counts in dict_val.values():
            yes = counts["Yes"]
            no = counts["No"]
            branch_n = yes + no
            weighted_gini += (branch_n / total_n) * giniFromCounts(yes, no)

        gini_split[feat] = weighted_gini

    best_feat = min(gini_split, key=gini_split.get)
    return best_feat, gini_split[best_feat]

def buildTree(df, features, eps=1e-12, depth=0, max_depth=4, min_samples_leaf=10):
    labels = df["Buys_Product"]

    # STOP 1: pure node
    if labels.nunique() == 1:
        return labels.iloc[0]

    # STOP 2: no features left
    if len(features) == 0:
        return labels.mode()[0]

    # STOP 3: depth limit
    if depth >= max_depth:
        return labels.mode()[0]

    # STOP 4: too few samples to make valid children
    # (must be able to create at least two leaves of size min_samples_leaf)
    if len(df) < 2 * min_samples_leaf:
        return labels.mode()[0]

    parent_g = nodeGini(df)

    # Choose best split
    best_feat, best_weighted = computeBestSplit(df, features)

    # STOP 5: no impurity improvement
    if best_weighted >= parent_g - eps:
        return labels.mode()[0]

    tree = {best_feat: {}}
    remaining_features = [f for f in features if f != best_feat]

    # Split and recurse
    for val in df[best_feat].unique():
        subset = df[df[best_feat] == val]

        # Enforce minimum leaf size
        if len(subset) < min_samples_leaf:
            tree[best_feat][val] = labels.mode()[0]
        else:
            tree[best_feat][val] = buildTree(
                subset,
                remaining_features,
                eps=eps,
                depth=depth + 1,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf
            )

    return tree


def treeToGraphviz(tree):
    dot = Digraph()
    node_id = 0

    def add_node(subtree):
        nonlocal node_id
        current_id = str(node_id)
        node_id += 1

        # Leaf
        if not isinstance(subtree, dict):
            dot.node(current_id, f"Predict: {subtree}", shape="box")
            return current_id

        # Internal node
        feature = next(iter(subtree))
        dot.node(current_id, feature)

        for value, child in subtree[feature].items():
            child_id = add_node(child)  # <-- fixed
            dot.edge(current_id, child_id, label=str(value))

        return current_id

    add_node(tree)
    return dot

tree = buildTree(dataset, features, max_depth=999, min_samples_leaf=1)
tree1 = buildTree(dataset1, features1, max_depth=4, min_samples_leaf=60)
tree2 = buildTree(dataset2, features2, max_depth=4, min_samples_leaf=60)
tree3 = buildTree(dataset3, features3, max_depth=4, min_samples_leaf=60)

def predict(tree, sample):
    
    while isinstance(tree, dict):
        feature = next(iter(tree))          
        value = sample[feature]
        tree = tree[feature][value]         
    return tree                             

dot = treeToGraphviz(tree)
dot1 = treeToGraphviz(tree1)
dot2 = treeToGraphviz(tree2)
dot3 = treeToGraphviz(tree3)

dot.render("decision_tree", format="png", cleanup=True)
dot1.render("decision_tree1", format="png", cleanup=True)
dot2.render("decision_tree2", format="png", cleanup=True)
dot3.render("decision_tree3", format="png", cleanup=True)

tree_pred = [predict(tree,row) for _, row in test.iterrows()]
tree_pred1 = [predict(tree1,row) for _, row in test1.iterrows()]
tree_pred2 = [predict(tree2,row) for _, row in test2.iterrows()]
tree_pred3 = [predict(tree3,row) for _, row in test3.iterrows()]

pred_df = pd.DataFrame(data={"Buys_Product": tree_pred})
pred_df1 = pd.DataFrame(data={"Buys_Product": tree_pred1})
pred_df2 = pd.DataFrame(data={"Buys_Product": tree_pred2})
pred_df3 = pd.DataFrame(data={"Buys_Product": tree_pred3})

pred_df.to_csv("./tree_pred.csv", sep=',',index=False)
pred_df1.to_csv("./tree_pred1.csv", sep=',',index=False)
pred_df2.to_csv("./tree_pred2.csv", sep=',',index=False)
pred_df3.to_csv("./tree_pred3.csv", sep=',',index=False)

def accuracy(preds, test_df):
    y_true = test_df["Buys_Product"].tolist()
    correct = sum(p == t for p, t in zip(preds, y_true))
    return correct / len(y_true)

acc  = accuracy(tree_pred, test)
acc1 = accuracy(tree_pred1, test1)
acc2 = accuracy(tree_pred2, test2)
acc3 = accuracy(tree_pred3, test3)

tree_acc = [acc,acc1,acc2,acc3]

pred_acc = pd.DataFrame(data={"Tree Accuracy": tree_acc})

pred_acc.to_csv("./tree_acc.csv", sep=',',index=False)
