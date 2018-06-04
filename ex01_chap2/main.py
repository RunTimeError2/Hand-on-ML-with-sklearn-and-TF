import os
import tarfile
from six.moves import urllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = 'datasets/housing'
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + '/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    print('Data already get.')


# fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# Using hash to generate the test set and train set
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing = load_housing_data()
#train_set, test_set = split_train_test(housing, 0.2)
#print(len(train_set), ' train + ', len(test_set), ' test')

#housing_with_id = housing.reset_index() # Adds an 'index' column
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')
#print(len(train_set), ' train + ', len(test_set), ' test')

housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    print('train_index=', train_index, ' test_index=', test_index)
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(housing['income_cat'].value_counts() / len(housing))
# Drop attribute income_cat
for set1 in (strat_train_set, strat_test_set):
    set1.drop(['income_cat'], axis=1, inplace=True)

# Using the train_set only
housing = strat_train_set.copy()
#housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
#             s=housing['population']/100, label='population',
#             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True
#             )
#plt.legend()
#plt.show()
#corr_matrix = housing.corr()
#print(corr_matrix['median_house_value'].sort_values(ascending=False))
#attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
#scatter_matrix(housing[attributes], figsize=(12, 8))
#plt.show()
#housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
#plt.show()

# split data and labels for training set
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

# Due to the fact that some value of the feature 'total_bedrooms' is missing, there are three options in total
# option1, getting rid of the corresponding districts
#housing.dropna(subset=['total_bedrooms'])
# option2, getting rid of the whole attribute
#housing.drop('total_bedrooms', axis=1)
# option3, set the values to some value(here uses the median value)
#median = housing['total_bedrooms'].median()
#housing['total_bedrooms'].fillna(median)

# Using the class Imputer to preprocess the data
imputer = Imputer(strategy='median') # create the imputer instance
housing_num = housing.drop('ocean_proximity', axis=1) # drop the non-numerical attribute
imputer.fit(housing_num) # the imputer instance calculates the median value of each attribute and fills in the missing values
# print the results, and the following two values should be the same
#print(imputer.statistics_)
#print(housing_num.median().values)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
#print(housing_tr)

# LabelEncoder is a transformer which converts test labels to numbers
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(housing_cat_encoded)
print(encoder.classes_)
# Such encoding method might bring about problems

# Using One-Hot encoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
print(housing_cat_1hot)
print(housing_cat_1hot.toarray())

# Using LabelBinarizer can combine the two process using LabelEncoder and OneHotEncoder
#encoder = LabelBinarizer(sparse_output=True) # The encoder will give a dense matrix if sparse_output=True does not exist
#housing_cat_1hot = encoder.fit_transform(housing_cat)
#print(housing_cat_1hot)


rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_estra_attribs = attr_adder.transform(housing.values)


# Using pipelines and featureunion
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)


num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', MyLabelBinarizer()),
    # Using LabelBinarizer() will cause errors due to the difference between version 0.18.0 and 0.19.0
    # In the new version the pipeline is assuming LabelBinarizer's fit_transform() method takes three positional arguments
    # like def fit_transform(self, x, y)
    # while it is defined to take only two: def fit_transform(self, x)
    # so that a new class MyLabelBinarizer should be defined to process the three arguments
])

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
print('housing_prepared:')
print(housing_prepared)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('Predictions:\n', lin_reg.predict(some_data_prepared))
print('Labels:\n', list(some_labels))

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print('Mean_square_root_error = ', lin_rmse, '\n')

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print('Mean_square_root_error of DecisionTreeRegressor = ', tree_rmse)
# This model seems to be working well, so a better evaluation method is needed


scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print('Scores = ', scores)
    print('Mean = ', scores.mean())
    print('Standard deviation = ', scores.std())


display_scores(tree_rmse_scores)
#lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
#                             scoring='neg_mean_squared_error', cv=10)
#lin_rmse_scores = np.sqrt(-lin_scores)
#display_scores(lin_rmse_scores)


forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(tree_mse)
print('Mean_square_root_error of ForestRegression = ', forest_rmse)


# Save the model
joblib.dump(forest_reg, 'forest_reg.pkl')
# Load the model
#forest_reg_loaded = joblib.load('forest_reg.pkl')


# Fine tuning the model with grid-search
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)


# Evaluate the importance of each feature
#feature_importances = grid_search.best_estimator_.feature_importances_
#print(feature_importances)

#extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
#cat_one_hot_attribs = list(encoder.classes_)
#attributes = num_attribs + extra_attribs + cat_one_hot_attribs
#print(sorted(zip(feature_importances, attributes), reverse=True))


# Evaluate the model on test set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print('Final_rmse = ', final_rmse)

print('Done.')
