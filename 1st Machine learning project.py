#!/usr/bin/env python
# coding: utf-8

# # AIM - BUILDING A MODEL FOR PREDICTING HOUSING PRICES IN CALIFORNIA USING THE CALIFORNIA CENSUS DATA

# # Importing libraries and housing.csv

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import os
import seaborn as sns 
housing = pd.read_csv('housing.csv')


# In[2]:


housing.head()


# Each row represents one district and there are 10 attributes i.e. columns

# In[3]:


housing.info()


# 1. Total_bedrooms - 20433,  Total records : 20640 but data available for only 20433. 
#    Missing data = 20640 - 20433 = 207
# 2. All column values are numerical except for 'ocean_proximity' which is string (categorical variable)

# In[4]:


#Shows how many different categories exist with total count of districts associated with each category

housing['ocean_proximity'].value_counts()


# In[5]:


# Describe method shows statistical summary of numerical attributes

housing.describe()

The count, mean, min and max rows are self-explanatory.

Note the count of total_bedrooms is 20,433, not 20,640. It means that null values are ignored

std rows shows the standard deviation (which measures how dispersed the values are)

25%, 50%, 75% shows the corresponding percentiles

Points to Note

25th percentile is called 1st quartile - 25% of the districts have a housing_median_age lower than 18.
50th percentile is called median - 50% of the districts have a housing_median_age lower than 29.
75th percentile is called 3rd quartile - 75% of the districts have a housing_median_age lower than 37.
# # PLOT HISTOGRAM ( Descriptive Analytics )

# In[6]:


# Let's plot a histogram to get the feel of type of data we are dealing with
# We can plot histogram only for numerical attributes

housing.hist(bins = 50, figsize = (20,15))
plt.show()


# In[7]:


# To make this notebook's output identical at every run

np.random.seed(42) # to fix the random permutation output
np.random.permutation(5)


# In[8]:


# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")


# In[9]:


# With unique and immutable identifier

import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

print(len(train_set), "train +", len(test_set), "test")


# In[10]:


test_set.head()


# In[11]:



# Combining latitude and longitude into an ID

housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

print(len(train_set), "train +", len(test_set), "test")

test_set.head()


# In[12]:


np.random.seed(2)
np.random.permutation(4)


# In[13]:


# SPLITTING THE DATASET INTO TRAIN & TEST DATASET

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

print(len(train_set), "train +", len(test_set), "test")

test_set.head()


# In[14]:


#random_state 'relevance' - eveytime we run the code test and train data will be the same provided 
# number we have used initially should be the same. This is done to prevent the algorithm to run on test data
# if we dont use it then, algorithm will run on test data as well in addition to the train data and we wont get desired results


# In[15]:


housing.hist()


# In[16]:


housing.median_income.hist()


# In[17]:


# limiting the categories in Median Income column by dividing the values/ 1.5 so that the spread is less
# Divide by 1.5 to limit the number of income categories
# Round up using ceil to have discrete categories

housing['income_cat'] = np.ceil(housing.median_income / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[18]:


housing.head()


# In[19]:


housing['income_cat'].value_counts()


# In[20]:


housing['income_cat'].hist()


# In[21]:


np.ceil([15.3,17.6]) # ceiling method converts the float to next highest integer


# In[22]:


# Stratified Sampling using Scikit-learn's StratifiedShuffleSplit Class

from sklearn.model_selection import StratifiedShuffleSplit


# In[23]:


split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state = 42)
# using StratifiedShuffleSplit to perform 1 split, give test data - 20% and resulting 80% will automatically be train data
# random_state = 42 is fixed to have fixed output so that test data is not included in train data


# In[24]:


for train_index,test_index in split.split(housing,housing.income_cat): # for loop and splitting the data 'housing' on the 
# basis of housing['income_cat'] column and creating stratified sample
    strat_train_set = housing.loc[train_index] # training set data (basis index) assigned to strat_train_set
    strat_test_set = housing.loc[test_index] # test set data (basis index) assigned to strat_test_set


# In[25]:


# Income category proportion in test set generated with stratified sampling
strat_test_set["income_cat"].value_counts() / len(strat_test_set)


# In[26]:


# Income category proportion in full dataset

housing["income_cat"].value_counts() / len(housing)


# # Lets find out the error % of Stratified sample & Random sample

# In[27]:


def income_cat_proportions(data):
    return data['income_cat'].value_counts()/len(data) # defined function to return 'income_cat' proportion in a dataset

train_set,test_set = train_test_split(housing,test_size = 0.2,random_state = 42) # splitting the housing data - 20% test, 80% training (automatically)

compare_props = pd.DataFrame({'Overall': income_cat_proportions(housing), 'Stratified':income_cat_proportions(strat_test_set),
                             'Random': income_cat_proportions(test_set)})
compare_props


# In[28]:


compare_props.sort_index(inplace = True) # sorting data by index
compare_props


# In[29]:


# creating a new column "Rand. %error" to save Random sampling / proportion error % data w.r.t Overall data (proportions)

compare_props["Rand. %error"] = 100 * ((compare_props["Random"] - compare_props["Overall"])/compare_props["Overall"])


# In[30]:


# creating a new column "Strat. %error" to save Stratified sampling / proportionerror % data w.r.t Overall data (proportions)

compare_props["Strat. %error"] = 100 *((compare_props["Stratified"]-compare_props["Overall"])/compare_props["Overall"])


# In[31]:


# we can see the error % for stratified proportion is less as compared to random so we should be considering the stratified 
# sampling for building the model

# values in the dataframe are the ratios under each category

compare_props


# In[32]:


# Dropping "income_cat" column from both the DF - training & test
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    


# # Discover and visualize the data to gain insights

# In[33]:


housing = strat_train_set.copy()


# In[34]:


housing.head()


# In[35]:


housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude')


# In[36]:


housing.plot(kind = 'scatter', x= 'longitude', y = 'latitude', alpha = 0.1)


# In[37]:


housing.plot(kind = 'scatter', x = 'longitude', y = 'latitude', alpha = 0.4, s = housing.population/100, label = 'population', 
            figsize = (10,7), c = 'median_house_value', cmap = plt.get_cmap('jet'), colorbar = True, sharex = False)
plt.legend()


# In[38]:


import matplotlib.image as mpimg
california_img=mpimg.imread('ml/machine_learning/images/end_to_end_project/california.png')
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                       s=housing['population']/100, label="Population",
                       c="median_house_value", cmap=plt.get_cmap("jet"),
                       colorbar=False, alpha=0.4,
                      )
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
plt.show()

After looking at the graph - 

1. Houses near coastal areas are expensive (red)
2. Not all houses near coastal areas are expensive
3. Need to find correlation why this is the case
# In[39]:


corr_matrix = housing.corr()
corr_matrix


# In[40]:


corr_matrix['median_house_value'].sort_values(ascending = False)


# In[41]:



# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas

from pandas.plotting import scatter_matrix
attributes = ['total_rooms','housing_median_age','median_income','median_house_value']
scatter_matrix(housing[attributes],figsize = (12,8))


# In[42]:


housing.plot(kind = 'scatter', y = 'median_house_value', x = 'median_income', alpha = 0.1)

Observations - 

1. Positive Correlation
2. Capping done - $500,000
3. Strange lines observed middle and upper left. We need to remove these in order to ensure appropriate working of algorithm
# In[43]:


housing.head(2)


# In[44]:


# Experimenting with Attribute Combinations - to find corrleation

housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing ['households']
housing.head(20)


# In[45]:


corr_matrix = housing.corr()
corr_matrix


# In[46]:


corr_matrix['median_house_value'].sort_values(ascending = False)


# In[47]:


housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()


# In[48]:


housing.describe()


# # Prepare the data for Machine Learning algorithms
# 

# In[49]:


# Let’s revert to a clean training set

housing = strat_train_set.drop('median_house_value',axis = 1)  # drop labels for training set
housing_labels = strat_train_set['median_house_value'].copy()
housing_labels

# Note drop() creates a copy of the data and does not affect strat_train_set


# In[50]:


isn = housing.isnull()
isn.any(axis=1)


# In[51]:


# Let’s experiment with sample dataset for data cleaning

sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head(100)
sample_incomplete_rows


# In[52]:


# Option one
# dropna() - drops the missing values / entire row w.r.t to missing values (Nan) in total_bedrooms column

sample_incomplete_rows.dropna(subset=["total_bedrooms"])


# In[53]:


# Option two
# drop() - drops the attribute

sample_incomplete_rows.drop('total_bedrooms',axis = 1)


# In[54]:


# Option three
# fillna() - sets the missing values
# Let’s fill the missing values with the median

median = housing['total_bedrooms'].median()
median


# In[55]:


sample_incomplete_rows.fillna(value=median, inplace = True)
sample_incomplete_rows


# In[56]:


# 58.41


# In[57]:


# Let's use Scikit-Learn Imputer class to fill missing values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')


# In[58]:


# Remove the text attribute because median can only be calculated on numerical attributes

housing_num = housing.drop('ocean_proximity',axis = 1)
housing_num.head(5)


# In[59]:


# Fit the imputer instance to the training data. fit method identiies the median for each column 
imputer.fit(housing_num)


# In[60]:


# code to know median computed for all the columns 
imputer.statistics_


# In[61]:


X = imputer.transform(housing_num) # transform method transforms the entire data by adding the median value
X


# In[62]:


housing_tr = pd.DataFrame(X,columns = housing_num.columns)
housing_tr


# In[63]:


# Convert ocean_proximity to numbers

housing_cat = housing.ocean_proximity
housing_cat


# In[64]:


# Pandas factorize() example

df = pd.DataFrame({
        'A':['type1','type3','type3', 'type2', 'type0','type3','type1']
    })
df


# In[65]:


df['A'].factorize()


# In[66]:


# Convert ocean_proximity to numbers

housing_cat.value_counts()


# In[67]:



# Convert ocean_proximity to numbers
# Use Pandas factorize()


housing_cat_encoded, housing_categories = housing_cat.factorize()
housing_cat_encoded[:10]


# In[68]:



# Check encoding classes

housing_categories


# In[69]:


# We can convert each categorical value to a one-hot vector using a `OneHotEncoder`
# Note that fit_transform() expects a 2D array
# but housing_cat_encoded is a 1D array, so we need to reshape it

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot


# In[70]:


housing_cat_1hot.toarray()


# In[71]:


# Just run this cell, or copy it to your code, do not try to understand it (yet).
# Definition of the CategoricalEncoder class, copied from PR #9151.

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


# In[72]:


# above code will do both tasks - factorization & then, one hot encoder


# In[73]:


# The CategoricalEncoder expects a 2D array containing one or more categorical input features. 
# We need to reshape `housing_cat` to a 2D array:

cat_encoder = CategoricalEncoder(encoding = 'onehot-dense')
housing_cat_reshaped = housing_cat.values.reshape(-1,1)
housing_cat_reshaped


# In[74]:


housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
housing_cat_1hot


# In[75]:


cat_encoder.categories_


# In[76]:



from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
housing_extra_attribs = pd.DataFrame(housing_extra_attribs, columns=list(housing.columns)+["rooms_per_household", "population_per_household"])
housing_extra_attribs.head()


# In[77]:


# Creating DataFrame first

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=(range(6)))
s2 = pd.Series([10, 9, 8, 7, 6, 5], index=(range(6)))
df = pd.DataFrame(s1, columns=['s1'])
df['s2'] = s2
df


# In[78]:


# Use Scikit-Learn minmax_scaling

from mlxtend.preprocessing import minmax_scaling
minmax_scaling(df, columns=['s1', 's2'])


# In[79]:


# Now let's build a pipeline for preprocessing the numerical attributes:

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[80]:


housing_num_tr


# In[81]:


from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


Now let's join all these components into a big pipeline that will preprocess both the numerical and the categorical features:
# In[82]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])


# In[83]:


num_pipeline.fit_transform(housing)


# In[84]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[85]:


housing.head()


# In[86]:



housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared[0])
print(housing_prepared[1])
print(housing_prepared[2])


# # Select and train a model

# In[87]:


# Train a Linear Regression model

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[88]:



# Let's try the full pipeline on a few training instances

some_data = housing.iloc[:5]
some_labels = housing_labels[:5]
some_data_prepared = full_pipeline.transform(some_data)


# In[89]:


print('Predictions', lin_reg.predict(some_data_prepared))


# In[90]:


# Print the actual values
print("Labels:", list(some_labels))


# In[91]:


# Calculate the RMSE in Linear Regression Model

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# In[92]:


# Train a model using Decision Tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# In[93]:


# Calculate RMSE in Decision Tree model

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# In[94]:


# Performs K-fold cross-validation
# Randomly splits the training set into 10 distinct subsets called folds
# Then it trains and evaluates the Decision Tree model 10 times By
# Picking a different fold for evaluation every time and training on the other 9 folds
# The result is an array containing the 10 evaluation scores

from sklearn.model_selection import cross_val_score

tree_reg = DecisionTreeRegressor(random_state=42)
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[95]:


# Look at the score of cross-validation of DecisionTreeRegressor

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)


# In[96]:


# Now compute the same score for Linear Regression

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)


# In[97]:


# Let's train one more model using Random Forests

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)


# In[98]:


# Calculate RMSE in Random Forest model

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


# In[99]:


# Cross Validation in Random Forest model

from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# # Fine-tune the Model

# In[100]:



param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

count = 0;
for i in [3, 10, 30]:
    for j in  [2, 4, 6, 8]:
        print(" I am goin to try %d <> %d" % (i, j) )
        count += 1

for k in [False]:
    for i in [3, 10]:
        for j in  [2, 3, 4]:
            print(" I am goin to try %d <> %d <> %s" % (i, j, k) )
            


# In[101]:



# GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


# In[102]:


# The best hyperparameter combinations

grid_search.best_params_


# In[103]:


# Get the best estimator

grid_search.best_estimator_


# In[104]:


# Let's look at the score of each hyperparameter combination tested during the grid search

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[105]:



from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)


# In[106]:


# See the importance score of each attribute in GridSearchCV

feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[107]:



# Evaluate model on the Test Set

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[108]:


final_rmse


# In[ ]:




