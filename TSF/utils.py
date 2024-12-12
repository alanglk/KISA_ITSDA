
from abc import ABC, abstractmethod
import pandas as pd

class TSInterface(ABC):
    # Time Series Interfacce for defining the methods that must
    # be implemented by other classes

    @abstractmethod
    def fit(X, y) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def predict():
        raise NotImplementedError

class M5Data():
    def __init__(
            self,
            calendar_path:str,
            sales_train_validation_path:str,
            sales_train_evaluation_path:str
            ):
        # date	wm_yr_wk    weekday	wday	month	year    d	event_name_1	event_type_1	event_name_2	event_type_2	snap_CA	snap_TX	snap_WI
        self.calendar           = pd.read_csv(calendar_path)
        self.calendar['date']   = pd.to_datetime(self.calendar['date'])

        # id    item_id dept_id cat_id  store_id    state_id    d_1	d_2	d_3	d_4	...	d_1913
        self.sales_train    = pd.read_csv(sales_train_validation_path) # Train data
        self.sales_test     = pd.read_csv(sales_train_evaluation_path) # Test data

        self.train_ts_prefix = "_validation" # prefix added to the Time Series id
        self.test_ts_prefix  = "_evaluation" # prefix added to the Time Series id

    def get_dates(self, df: pd.DataFrame, real = True):
        if not real:
            return list(map(int, [col.split('_')[1] for col in df.columns] )) 
        return self.calendar.set_index('d').loc[df.columns, 'date']

    def get_train_val_ts(self, 
                         ts_id:str = "HOBBIES_1_001_CA_1", 
                         n_splits: int = 3, 
                         n_days: int = 28, 
                         dynamic_start_pos: bool = False):
        """
        Get the train and dev sets with increasing cross-validation.
        ### INPUT
            - ``ts_id``: Time Series item_id
            - ``n_splits``: Number of cross validation splits (3 by default).
            - ``n_days``: Number of hold-out days of each split (28 by default).
            - ``dynamic_start_pos``: Wheter to start where the time series is not 0 or to start at d_1 (2011-01-29).
        ### OUTPUT 
            ``train_data``, ``ground_truth_data``
        """
        
        # With default params:
        # Tree cross-validation splits
        #   - cv1 : d_1830 ~ d_1857
        #   - cv2 : d_1858 ~ d_1885
        #   - cv3 : d_1886 ~ d_1913  

        ts_id = ts_id + self.train_ts_prefix 
        ts = self.sales_train[self.sales_train['id'] == ts_id]
        N = len(ts.columns)
        train_data  = []
        cv_data     = []
        
        start_train = ts.columns.get_loc('d_1') # d_1 position
        if dynamic_start_pos:
            # Serch for the first position where there are any sales
            for col in ts.columns[start_train:]:
                if ts[col].iloc[0] != 0: 
                    dyn_index = ts.columns.get_loc(col)
                    break

            assert dyn_index >= start_train
            start_train = dyn_index
        
        # Check that there are entries for the train set
        assert start_train < N - n_days * n_splits 

        for k in reversed(range(n_splits)):
            start_cv    = N - n_days * (k+1)
            ts_train= ts.iloc[:, start_train:(start_cv -1)]
            ts_cv   = ts.iloc[:, start_cv:(start_cv + n_days)]
                        
            train_data.append(ts_train)
            cv_data.append(ts_cv)
        
        return train_data, cv_data
    
    def get_test_ts(self, ts_id:str = "HOBBIES_1_001_CA_1_evaluation"):
        raise NotImplementedError

