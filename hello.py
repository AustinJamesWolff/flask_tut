from flask import Flask, request
import pickle

app = Flask(__name__)

clf = pickle.load(open('model.pkl','rb'))

@app.route("/")
def hello():
    return "Hello World!"

def get_model():
    from sklearn.datasets import load_iris 
    from sklearn.linear_model import LogisticRegression 
    import pickle
    X, y = load_iris(return_X_y = True)
    clf = LogisticRegression(random_state=0, solver='lbfgs',
                             multi_class='multinomial').fit(X, y)
    clf_string = pickle.dumps(clf)
    return clf_string, X, y


@app.route("/iris")
def iris():
    from sklearn.datasets import load_iris 
    X, y = load_iris(return_X_y = True)
    # from sklearn.linear_model import LogisticRegression 
    # import pickle

    # clf_string, X, y = get_model()
    # clf = pickle.loads(clf_string)

    # clf = pickle.load(open('model.pkl','rb'))

    return str(clf.predict(X[:2, :]))

@app.route("/score")
def score():
    import pickle
    clf_string, X, y = get_model()
    clf = pickle.loads(clf_string)
    return str(clf.score(X, y))