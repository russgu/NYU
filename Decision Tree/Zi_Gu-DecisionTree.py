
#Talked to Meet Barot

import numpy as np
from collections import Counter
import itertools
from sklearn import tree, grid_search
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO with open('decision_tree.dot', 'w') as f:



train_text=open('adult_train.txt');
train=train_text.read().split("\n");
test_text=open('adult_test.txt');
test=test_text.read().split("\n");
features_text=open('features.txt');
features=features_text.read().split("\n");
result=[];


modify(train);
Handle_MissingValues(train);
result=transform_feature(train);


def modify(train):
    j=0;
    for vector in train:
        array=vector.split(",");
        train[j]=array;
        j+=1;


#Part a)
def Handle_MissingValues(train):
    for i in range(0,12):
        list=[];
        sum=0;
        for vector in train:
            if vector[i].isdigit():
                sum+=int(vector[i]);
            elif vector[i]!=" ?": 
                list.append(vector[i]);
        data = Counter(list);
        category_value=data.most_common(1);
        if len(category_value)==1:
             category_value=category_value[0][0];
        continuous_value=sum/len(train)*1.0;
       
        for v in train:
            if v[i]==" ?":
                if i>=8 and i<=10 or i==0:
                    v[i]=continuous_value;
                else :
                    v[i]=category_value;
                      

#Part b)
def transform_feature(train):
    classification=[];
    dictionary=[];
    for x in features:
        array=x.split(":");
        array[1]=array[1].split(",");
        dictionary.append(array);
        
    j=0;
    for x in train:
        for feat,y in zip(dictionary,range(len(x))):
            new_features=[];
            string=x[y];
            if string.isdigit()==True:
                new_features.append(int(string));
            elif string.isdigit()==False:
                for z in feat[1]:
                    new_feat_name=feat[0]+"_"+z;
                    if string in new_feat_name:
                        new_features.append(1);
                    else:
                        new_features.append(0);
            x[y]=new_features;
        if x[-1]==" >50K" or x[-1]==" >50K.":
            classification.append(1); #the class for each featuer vector
        else:
            classification.append(0);
        x = list(itertools.chain(*x[:-1]));
        train[j]=x;
        j+=1;
        
    return classification;
        

#Part c)

def DecisionTreeClassifier():
    depth=[x for x in range(1,31)];
    leaf=[];
    for i in range(1,51):
        leaf.append(i);
    parameter_depth = {'max_depth' : depth}
    parameter_leaf={'min_samples_leaf': leaf}
    
    
    training_Y=result[:22793];
    training_X=train[:22793];
    validate_Y=result[22794:];
    validate_X=train[22794:];
    
    clf = tree.DecisionTreeClassifier();
    grid1=grid_search.GridSearchCV(clf, parameter_depth, scoring='accuracy');
    grid2=grid_search.GridSearchCV(clf, parameter_leaf, scoring='accuracy');
    
    grid1.fit(training_X, training_Y);
    score_depth=grid1.grid_scores_;
    accuracy_training=[result.mean_validation_score for result in score_depth];
    grid1.fit(validate_X,validate_Y);
    score_val_depth=grid1.grid_scores_;
    accuracy_validate=[result.mean_validation_score for result in score_val_depth];
    
    plt.plot(depth,accuracy_training);
    plt.plot(depth,accuracy_validate);
    plt.xlabel('Depth');
    plt.ylabel('Accuracy');
    plt.show();

    
    grid2.fit(training_X, training_Y);
    score_leaf=grid2.grid_scores_;
    accuracy_training_leaf=[result.mean_validation_score for result in score_leaf];
    grid2.fit(validate_X,validate_Y);
    score_val_leaf=grid2.grid_scores_;
    accuracy_validate_leaf=[result.mean_validation_score for result in score_val_leaf];
    
    plt.plot(leaf,accuracy_training_leaf);
    plt.plot(leaf,accuracy_validate_leaf);
    plt.xlabel('Leaf');
    plt.ylabel('Accuracy');
    plt.show();
    
   
    f = tree.export_graphviz(clf, out_file=f, max_depth=2);

    

DecisionTreeClassifier();  

        
#Part d)
modify(test);
Handle_MissingValues(test);
result2=transform_feature(test);


clf = tree.DecisionTreeClassifier();
clf = clf.fit(train,result);
error=0;
array=clf.predict(test).tolist();

for x,y in zip(result2,array):
    if x!=y:
        error+=1;
length=len(array);
avg=error*1.0/length;

print"The error for test set is: "
print avg;

                
                
    




