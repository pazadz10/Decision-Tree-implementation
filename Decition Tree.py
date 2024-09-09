import math
import pandas as pd
import pptree
import numpy as np
import os

file_path = os.path.join(os.getcwd(), 'Smoking.csv')
df = pd.read_csv(file_path)


def build_tree(ratio):
    file_path = os.path.join(os.getcwd(), 'Smoking.csv')
    df = pd.read_csv(file_path)
    df = bucket(df)
    df = df.drop('ID', axis=1)
    div = np.random.rand(len(df)) < ratio
    data = df[div]
    validate = df[~div]
    smoking = df["smoking"]
    tree = Node(attribute='start', value='None')
    tree = tree.Learn_Decision_Tree(data=data, Attribute=df.columns)
    pptree.print_tree(tree)
    error = tree.error_rate(validate)
    print(f'the error of the tree is {round(error, 2)}')




def tree_error(k):
    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    file_path = os.path.join(os.getcwd(), 'Smoking.csv')
    df = pd.read_csv(file_path)
    df.drop(columns="ID")
    df = bucket(df)
    error = 0
    for i in range(1, k):
        ratio = (i / k) * len(df)
        examples = df[:int(ratio)]
        toValidate = df[int(ratio):]
        tree = Node(attribute='start', value='None')
        tree = tree.Learn_Decision_Tree(data=examples, Attribute=examples.columns)
        Kerror = tree.error_rate(toValidate)
        error += Kerror
    print(f'the average error of {k}-fold cross validation model is {error / k}')

    return error / (k - 1)



def is_busy(rowInput):  # gets expected value from single raw
    file_path = os.path.join(os.getcwd(), 'Smoking.csv')
    AllExamples = pd.read_csv(file_path)
    df = pd.DataFrame(columns=AllExamples.columns)
    rowInput=pd.DataFrame([rowInput], columns=AllExamples.columns)
    df = pd.concat([df, rowInput], ignore_index=True)
    df = df.drop('ID', axis=1)
    AllExamples = AllExamples.drop('ID', axis=1)
    AllExamples= bucket(AllExamples)
    example = bucket(df)
    test = example
    tree = Node(attribute='start', value='None')
    tree = tree.Learn_Decision_Tree(data=AllExamples, Attribute=AllExamples.columns)

    check =tree.rowCheck(test)
    if check == 1:
        print("smoking, Please Stop it's not healthy")
    else:
        print("Not Smoking")


def bucket(df):
    df['age'] = df['age'].apply(lambda i: 0 if i < 44 else 1)
    df['height(cm)'] = df['height(cm)'].apply(lambda i: 0 if i < 164 else 1)
    df['weight(kg)'] = df['weight(kg)'].apply(lambda i: 0 if i < 65.8 else 1)
    df['waist(cm)'] = df['waist(cm)'].apply(lambda i: 0 if i < 82 else 1)
    df['eyesight(left)'] = df['eyesight(left)'].apply(lambda i: 0 if i < 1.012 else 1)
    df['eyesight(right)'] = df['eyesight(right)'].apply(lambda i: 0 if i < 1.007443 else 1)
    df['hearing(left)'] = df['hearing(left)'].apply(lambda i: 0 if i == 1 else 1)
    df['hearing(right)'] = df['hearing(right)'].apply(lambda i: 0 if i == 1 else 1)
    df['systolic'] = df['systolic'].apply(lambda i: 0 if i < 121.494218 else 1)
    df['relaxation'] =  df['relaxation'].apply(lambda i: 0 if i < 76.004830 else 1)
    df['fasting blood sugar'] =  df['fasting blood sugar'].apply(lambda i: 0 if i < 99.312325 else 1)
    df['Cholesterol'] = df['Cholesterol'].apply(lambda i: 0 if i < 196.901422 else 1)
    df['triglyceride'] = df['triglyceride'].apply(lambda i: 0 if i < 126.665697 else 1)
    df['HDL'] = df['HDL'].apply(lambda i: 0 if i < 57.290347 else 1)
    df['LDL'] =  df['LDL'].apply(lambda i: 0 if i < 114.964501 else 1)
    df['hemoglobin'] = df['hemoglobin'].apply(lambda i: 0 if i < 14.622592 else 1)
    df['Urine protein'] =  df['Urine protein'].apply(lambda i: 0 if i < 1.087212 else 1)
    df['serum creatinine'] =  df['serum creatinine'].apply(lambda i: 0 if i < 0.885738 else 1)
    df['AST'] =df['AST'].apply(lambda i: 0 if i < 26.182935 else 1)
    df['ALT'] = df['ALT'].apply(lambda i: 0 if i < 27.036037 else 1)
    df['Gtp'] = df['Gtp'].apply(lambda i: 0 if i < 39.952201 else 1)
    df['tartar'] =  df['tartar'].apply(lambda i: 0 if i == 'N' else 1)
    df['gender'] = df['gender'].apply(lambda i: 0 if i == 'M' else 1)



    return df

class Node:
    def __init__(self, label=None, leaf=False, attribute=None, value=None):
        self.attribute = attribute
        self.children = []
        self.label = label
        self.leaf = leaf
        self.root = None
        self.onlyLeafs = []
        if leaf:

            self.value = value



    def __str__(self):
        print = ''
        if not self.leaf:
            return print + '--' + self.attribute
        elif self.value == 1:
            return '---> smoking'
        else:
            return '---> Not smoking'

    def Learn_Decision_Tree(self, data=None, Attribute=None, parent_data=None):
        if len(data) == 0:  # if no examples pick pluralty value from parent to leaf node
            return Node(value=parent_data['smoking'].mode(), leaf=True)
        elif len(Attribute) == 1:  # if there is only one attribute left, pick most common value to leaf node
            return Node(value=data['smoking'].mode(), leaf=True)
        elif len(data['smoking'].unique()) == 1:  # if smoking col contain 1 value
            return Node(value=data['smoking'].mode(), leaf=True)
        else:
            best_attribute = self.bestAttribute(data) # find most importent attribute
            leftTree= data.loc[data[best_attribute] == 0]
            rightTree = data.loc[data[best_attribute] == 1]
            leftTree = leftTree.drop(columns=[best_attribute])
            rightTree = rightTree.drop(columns=[best_attribute])

            Attribute.drop(best_attribute)

            tree = Node(attribute=best_attribute, leaf=False)  # create new subtree

            subTreeLeft = self.Learn_Decision_Tree(leftTree, leftTree.columns,parent_data=data)  # recursive call on new node examples
            subTreeLeft.label = 0
            subTreeLeft.value = 0
            subTreeRight = self.Learn_Decision_Tree(rightTree, rightTree.columns, parent_data=data)  # recursive call on new node examples
            subTreeRight.label = 1
            subTreeRight.value = 1
            subTreeLeft.value_count = leftTree['smoking'].value_counts()  # count subree true/false
            tree.children.append(subTreeLeft)
            subTreeRight.value_count = rightTree['smoking'].value_counts()  # count subree true/false
            tree.children.append(subTreeRight)
            if self.attribute == 'start':
                self.children.append(tree)
                self.label =1
                self.value =1
            return tree
    def bestAttribute(self, data):
        Attributes = data.columns
        Attributes = Attributes.drop('smoking')
        bestEntVal =1000
        best_attribut="s"
        for attribute in Attributes:
            ent=self.calculateEntropy(data,attribute)
            if ent < bestEntVal:
                best_attribut =attribute
        return best_attribut
    def calculateEntropy(self, data, attribute):
        column1 = data[attribute]
        smoking = data["smoking"]
        leftTrue = sum((column1 == 0) & (smoking == 1))
        rightTrue = sum((column1 == 1) & (smoking == 1))
        leftFalse = sum((column1 == 0) & (smoking == 0))
        rightFalse = sum((column1 == 1) & (smoking == 0))
        probLeft = sum(column1 == 0) / sum(column1 != 9)
        probRight = sum(column1 == 1) / sum(column1 != 9)
        entropy11 = 0
        entropy22 = 0

        if (leftTrue + leftFalse) != 0:
           falseSmoking = leftTrue / (leftTrue + leftFalse)
           falseNotSmoking = 1-falseSmoking
           if falseSmoking < 1 and falseSmoking > 0:
              entropy11 = -(falseSmoking * math.log2(falseSmoking) + falseNotSmoking * math.log2(falseNotSmoking))
           else: entropy11 =0

        if (rightFalse+rightTrue) != 0:
            trueSmoking = rightTrue / (rightTrue + rightFalse)
            trueNotSmoking = 1-trueSmoking
            if trueSmoking < 1 and trueSmoking > 0:
                entropy22 = -(trueSmoking * math.log2(trueSmoking) + trueNotSmoking * math.log2(trueNotSmoking))
            else:
                entropy22 = 0

        entValue = probLeft*entropy11+probRight*entropy22
        return entValue
    def Validation(self, example, flag = None): # validate example from decision tree created
        Root = self
        while Root.leaf == False:
            value_match = False;
            for child in Root.children:
                x= example[Root.attribute]
                if child.value == example[Root.attribute]:
                    Root = child
                    if flag:
                        return value_match
                    value_match = True
                    break
            if flag:
                return value_match
            if value_match == False:
                return Root.value_count.idxmax()
        return Root.label
    def error_rate(self, toValidate):
        error = 0
        for index, example in toValidate.iterrows(): # Validate each row from test data
            ans = self.Validation(example)
            if example['smoking'] != ans:
                error += 1

        return error / len(toValidate)
    def rowCheck(self, toValidate):
        error = 0
        ans = 0
        for index, example in toValidate.iterrows(): # Validate each row from test data
            ans = self.Validation(example)


        return ans




build_tree(0.005)
tree_error(5)
rowInput = [2,'M',55,170,60,80,0.8,0.8,1,1,138,86,89,242,182,55,151,15.8,1,1,21,16,22,0,'N',1]

is_busy(rowInput)









