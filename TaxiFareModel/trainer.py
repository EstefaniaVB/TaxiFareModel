from TaxiFareModel.data import clean_data, get_data, holdout
from TaxiFareModel.utils import compute_rmse
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder

class Trainer():
    def __init__(self,X,y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test=None
        self.y_train =None
        self.y_test  = None
        
    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())
                              ])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))
                              ])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
                                          ('time', time_pipe, ['pickup_datetime'])], 
                                         remainder="drop")
        self.pipeline = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', LinearRegression())
                         ])

    def run(self):
        """set and train the pipeline"""
        X_train, X_test, y_train, y_test = holdout(self.X,self.y)
        self.set_pipeline()
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        print(rmse)
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)
    # train
    trainer= Trainer(X,y)
    trainer.run()
    X_train, X_test, y_train, y_test = holdout(X,y)
    # evaluate
    rmse =trainer.evaluate(X_test,y_test)
    print('rmse')
