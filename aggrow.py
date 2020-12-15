# Imports section
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier


# ML Algorithm setup
isTrained = False
dt_cls = DecisionTreeClassifier(max_depth=8)
svc_cls = SVC(gamma='auto', probability=True)
gb_cls = GradientBoostingClassifier(n_estimators=20, learning_rate=0.25,
                                     max_features=3, max_depth=2,
                                     random_state=0)
vote_cls = VotingClassifier(estimators=[('SVC', svc_cls),
                                         ('DTree', dt_cls),
                                         ('BoostingReg', gb_cls)], voting='soft')


# Dataset fetch & setup
data = pd.read_csv("Dataset.csv")
global crops_list
crops_list = data['crop'].unique()
x = data.drop('crop', axis=1)
y = data['crop']


# Model train
vote_cls.fit(x, y)

# User input
user_ip = [[0, 0, 0]]

# Predict
s = {}
s['rainfall'] = input('Enter rainfall in mm')
s['temperature'] = input('Enter temperature in C')
s['pH'] = input('Enter pH')
user_ip[0][0] = s['rainfall']
user_ip[0][1] = s['temperature']
user_ip[0][2] = s['pH']
global vote_cls
vote_result = vote_cls.predict_proba(user_ip)[0]
global crops_list
probabilities = pd.DataFrame(list(zip(vote_result, crops_list)), columns =['vote', 'crops'])
probabilities.sort_values("vote", axis = 0, ascending = False, inplace = True, na_position ='last')
predictions = list(probabilities['crops'].head(3))
print (predictions)
