import sklearn
import smt.surrogate_models as smtmodels
from pyearth import Earth
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_models(x_train, y_train):
    models = [smtmodels.KRG(corr='abs_exp', print_global=False), smtmodels.RBF(poly_degree=1, print_global=False),
                   DecisionTreeRegressor(), Earth(), SVR()]
    for model in models:
        if smtmodels.__name__ in getattr(model, '__module__', None):
            model.set_training_values(x_train, y_train)
            model.train()
        elif 'sklearn' in getattr(model, '__module__', None) or 'pyearth' in getattr(model, '__module__', None):
            model.fit(x_train, y_train)
    return models


def predict_models(models, x_test):
    pred = []
    for model in models:
        if smtmodels.__name__ in getattr(model, '__module__', None):
            pred.append(model.predict_values(x_test).flatten())
        elif 'sklearn' in getattr(model, '__module__', None) or 'pyearth' in getattr(model, '__module__', None):
            pred.append(model.predict(x_test).flatten())
    return pred