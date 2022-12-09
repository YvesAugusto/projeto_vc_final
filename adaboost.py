from src.util.dataset import load_dataset
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split, cross_validate
from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

data, target = load_dataset()

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2)

pipeline = Pipeline(
    [
        ('scaler', MinMaxScaler()), 
        ('decision_tree', AdaBoostClassifier()),
    ]
)

parameters = {
    'decision_tree__n_estimators': [3, 5, 10, 15, 20, 30, 40, 50],
}

search = GridSearchCV(pipeline, parameters, cv=5)
search.fit(X_train, Y_train)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

best_params = {}
for key, value in search.best_params_.items():
    key_ = key.split("__")[-1]
    best_params[key_] = value

scoring = {
    'f1': make_scorer(f1_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score),
    'accuracy': make_scorer(accuracy_score)
}

decision_tree = AdaBoostClassifier(**best_params)

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

cv_results = cross_validate(decision_tree, data, target, cv=cv,
                            scoring=scoring)

print(
    cv_results['test_accuracy'].mean(), cv_results['test_accuracy'].std(),
    cv_results['test_recall'].mean(), cv_results['test_recall'].std(),
    cv_results['test_precision'].mean(), cv_results['test_precision'].std(),
    cv_results['test_f1'].mean(), cv_results['test_f1'].std()
)