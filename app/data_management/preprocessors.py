
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import label_binarize



class ColumnSelector(BaseEstimator, TransformerMixin):
    """Retain or drop specified columns."""
    def __init__(self, columns, selector_type='keep'):
        self.columns = columns
        assert selector_type in ["keep", "drop"]
        self.selector_type = selector_type        
        
    def fit(self, X, y=None):
        return self    
    
    def transform(self, X):  
        if self.selector_type == 'keep':
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == 'drop':
            dropped_cols = [col for col in X.columns if col in self.columns]  
            X = X.drop(dropped_cols, axis=1)     
        return X


class ValueClipper(BaseEstimator, TransformerMixin): 
    def __init__(self, fields_to_clip, min_val, max_val) -> None:
        super().__init__()
        self.fields_to_clip = fields_to_clip
        self.min_val = min_val
        self.max_val = max_val
    
    def fit(self, data): return self
    
    def transform(self, data): 
        for field in self.fields_to_clip:
            if self.min_val is not None: 
                data[field] = data[field].clip(lower=self.min_val)
            if self.max_val is not None: 
                data[field] = data[field].clip(upper=self.max_val)
        return data




class MostFrequentImputer(BaseEstimator, TransformerMixin):  
    def __init__(self, cat_vars, threshold): 
        self.cat_vars = cat_vars
        self.threshold = threshold
        self.fill_vals = {}
    
    def fit(self, X, y=None):  
        self.fitted_cat_vars = [ 
            var for var in self.cat_vars
            if var in X.columns and X[var].isnull().mean() <  self.threshold ]

        for col in self.fitted_cat_vars: 
            self.fill_vals[col] = X[col].value_counts().index[0] 
        return self
    

    def transform(self, X, y=None):
        for col in self.fill_vals: 
            if col in X.columns: 
                X[col] = X[col].fillna(self.fill_vals[col])
        return X


class OneHotEncoderMultipleCols(BaseEstimator, TransformerMixin):  
    def __init__(self, ohe_columns, max_num_categories=10): 
        super().__init__()
        self.ohe_columns = ohe_columns
        self.max_num_categories = max_num_categories
        self.top_cat_by_ohe_col = {}
        
        
    def fit(self, X, y=None):    
        for col in self.ohe_columns:
            if col in X.columns: 
                self.top_cat_by_ohe_col[col] = [ 
                    cat for cat in X[col].value_counts()\
                        .sort_values(ascending = False).head(self.max_num_categories).index
                    ]         
        return self
    
    
    def transform(self, data): 
        data.reset_index(inplace=True, drop=True)
        df_list = [data]
        cols_list = list(data.columns)
        for col in self.ohe_columns:
            if len(self.top_cat_by_ohe_col[col]) > 0:
                if col in data.columns:                
                    for cat in self.top_cat_by_ohe_col[col]:
                        col_name = col + '_' + cat
                        vals = np.where(data[col] == cat, 1, 0)
                        df = pd.DataFrame(vals, columns=[col_name])
                        df_list.append(df)
                        
                        cols_list.append(col_name)
                else: 
                    raise Exception(f'''
                        Error: Fitted one-hot-encoded column {col}
                        does not exist in dataframe given for transformation.
                        This will result in a shape mismatch for train/prediction job. 
                        ''')
        transformed_data = pd.concat(df_list, axis=1, ignore_index=True) 
        transformed_data.columns =  cols_list
        return transformed_data


class CustomLabelBinarizer(BaseEstimator, TransformerMixin): 
    def __init__(self, target_field, target_class) -> None:
        super().__init__()
        self.target_field = target_field
        self.target_class = target_class
        self.given_classes = None

    def fit(self, data):         
        # grab the two classes
        given_classes = data[self.target_field].drop_duplicates().tolist()
        # sort so that the target class is last
        given_classes.sort(key = lambda k: k == self.target_class)
        # save for transformation
        self.given_classes = given_classes
        return self     
    
    def transform(self, data):
        if self.target_field in data.columns: 
            data[self.target_field] = label_binarize(
                data[self.target_field], 
                classes = self.given_classes
            ).flatten()
        return data
    