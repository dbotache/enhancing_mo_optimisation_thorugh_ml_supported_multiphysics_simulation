from sklearn.model_selection import GridSearchCV


class ParameterSearch:

    def __init__(self, X, y, model, parameters, scoring=None, cv=10, n_jobs=1):
        
        self.X = X
        self.y = y
        self.model = model
        self.parameters = parameters
            
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring
        
        self.best_estimator = None
        self.best_params = None

    def grid_search(self):
        
        grid_search = GridSearchCV(self.model, self.parameters, scoring=self.scoring, cv=self.cv, n_jobs=self.n_jobs)
        grid_search.fit(self.X, self.y)
        
        return grid_search.best_params_, grid_search.best_estimator_


def tune_params(X, y, model, name, params, scoring=None, cv=10, n_jobs=1):
    
    if scoring is None:
        scoring = 'neg_mean_squared_error'

    ps = ParameterSearch(X, y, model, params, scoring=scoring, cv=cv, n_jobs=n_jobs)
    best_parameters, best_estimator = ps.grid_search()

    return best_parameters, best_estimator