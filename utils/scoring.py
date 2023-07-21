from sklearn.base import clone
import numpy as np

def weighted_cross_val_score(model, X, y, 
                            groups=None, cv=None,
                            weights=None, scoring=None, 
                            use_weights_to_fit=True):
    scores = []
    for train_index, test_index in cv.split(X, y, groups=groups):
        model_clone = clone(model)

        # This is really bad code but does the job for this project
        if isinstance(X, np.ndarray):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        else:
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        if weights is not None:
            weights_train, weights_test = weights[train_index], weights[test_index]

        if weights is not None and use_weights_to_fit:
            try:
                model_clone.fit(X_train, y_train,sample_weight=weights_train)
            except:
                model_clone.fit(X_train, y_train, model__sample_weight=weights_train)
        else:
            model_clone.fit(X_train, y_train)
            
        y_pred = model_clone.predict(X_test)
        
        score = scoring(y_test, y_pred, sample_weight=weights_test)
        scores.append(score)
    return scores