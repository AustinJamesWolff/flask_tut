from sklearn.datasets import load_iris 
from sklearn.linear_model import LogisticRegression 
import pickle
import requests
import json
X, y = load_iris(return_X_y = True)
clf = LogisticRegression(random_state=0, solver='lbfgs',
                            multi_class='multinomial').fit(X, y)
# clf_string = pickle.dumps(clf)
pickle.dump(clf, open('model.pkl','wb'))
# return clf_string, X, y