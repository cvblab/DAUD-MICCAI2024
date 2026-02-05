# Processing data

import pandas as pd
import collections
from utils import *

class data_partition:
    def __init__(self, CSV_file, config=0, unsupervised=True):

        # Read & Load data
        self.data = self._read_data(CSV_file)
        # Create test data
        self.A_test, self.B_test = self.datatest(self.data)

        # Train
        self.A_train = self.datatrain(self.data, self.A_test, 'A', unsupervised)
        self.B_train = self.datatrain(self.data, self.B_test, 'B', unsupervised)


        self.train_config, self.dom_test_A, self.dom_test_B  = self.configuration(self.A_train, self.B_train, config)
        

    def configuration(self, A, B, config):
        train_config = {}

        if config == 1: # Train from Vlc and adding percentage of Grx samples
            train_config['values'] = A['values']
            train_config['labels'] = A['labels']
            # Training to add
            train_config['perc_values'] = B['values']
            train_config['perc_labels'] = B['labels']
            
            dom_test_A, dom_test_B = torch.ones(len(self.A_test['dataset'])), torch.zeros(len(self.B_test['dataset']))

        else: # Train from Grx and adding percentage of Vlc samples
            train_config['values'] = B['values']
            train_config['labels'] = B['labels']
            # Training to add
            train_config['perc_values'] = A['values']
            train_config['perc_labels'] = A['labels']

            dom_test_A, dom_test_B = torch.zeros(len(self.A_test['dataset'])), torch.ones(len(self.B_test['dataset']))

        return train_config, dom_test_A, dom_test_B
            
    def _read_data(self, CSV_file):
        data = pd.read_csv(CSV_file, delimiter=",")
        data['outlier'] = data['GT'].apply(lambda x: 0 if x == 0 or x == 2 else 1)
        return data
    
    def datatest(self, data):
        # Define dictionaries
        A = {}
        B = {}

        ################## A ##########################
        # Select 18 samples from each of the two classes with outlier=0 from A
        A_outlier_0 = data[(data['WSI'].str.startswith('A')) & (data['outlier'] == 0)].groupby('GT').apply(lambda x: x.sample(n=18, random_state=2311))
        # Select 9 samples from each of the 4 classes with outlier=1 from A
        A_outlier_1 = data[(data['WSI'].str.startswith('A')) & (data['outlier'] == 1)].groupby('GT').apply(lambda x: x.sample(n=9, random_state=2311))
        
        # Combine the datasets
        A['dataset'] = pd.concat([A_outlier_0, A_outlier_1])
        A['values'] = A['dataset'][A['dataset']['WSI'].str.startswith('A')]['WSI'].values
        A['labels'] = A['dataset'][A['dataset']['WSI'].str.startswith('A')]['outlier'].values
        A['classes'] = A['dataset'][A['dataset']['WSI'].str.startswith('A')]['GT'].values

        ################## B ##########################
        # Select 28 samples from each of the 2 classes with outlier=0 from B
        B_outlier_0 = data[(data['WSI'].str.startswith('B')) & (data['outlier'] == 0)].groupby('GT').apply(lambda x: x.sample(n=28, random_state=42))
        # Select 14 samples from each of the 4 classes with outlier=1 from B
        B_outlier_1 = data[(data['WSI'].str.startswith('B')) & (data['outlier'] == 1)].groupby('GT').apply(lambda x: x.sample(n=14, random_state=42))
    
        # Combine the datasets
        B['dataset'] = pd.concat([B_outlier_0, B_outlier_1])
        B['values'] = B['dataset'][B['dataset']['WSI'].str.startswith('B')]['WSI'].values
        B['labels'] = B['dataset'][B['dataset']['WSI'].str.startswith('B')]['outlier'].values
        B['classes'] = B['dataset'][B['dataset']['WSI'].str.startswith('B')]['GT'].values

        # Print
        print_nbs(A_outlier_1, A_outlier_0, B_outlier_1, B_outlier_0)
        print('B distribution: ', collections.Counter(B['classes']))
        print('A distribution: ', collections.Counter(A['classes']))

        return A, B
    
    def datatrain(self, data, test, hosp, unsupervised):
        dict_data = {}
        test = test['dataset']
        train = data[~data['WSI'].isin(test['WSI'])]
        values_train = train[train['WSI'].str.startswith(hosp)]['WSI'].values
        labels_train = train[train['WSI'].str.startswith(hosp)]['outlier'].values
        # Ensure that the training data is normal
        if unsupervised:
            dict_data['values'] = values_train[labels_train==0]
            dict_data['labels'] = labels_train[labels_train==0]
        else:
            dict_data['values'] = values_train
            dict_data['labels'] = labels_train
        return dict_data
    

class data_partition_A_B_ood:
    def __init__(self, CSV_file, config=0, unsupervised=True):

        # Read & Load data
        self.data = self._read_data(CSV_file)
        # Create test data
        self.A_test, self.B_test = self.datatest(self.data)

        # Train
        self.A_train = self.datatrain(self.data, self.A_test, 'A', unsupervised)
        self.B_train = self.datatrain(self.data, self.B_test, 'B', unsupervised)
            
    def _read_data(self, CSV_file):
        data = pd.read_csv(CSV_file, delimiter=",")
        data['outlier'] = data['GT'].apply(lambda x: 0 if x == 0 or x == 2 else 1)
        return data
    
    def datatest(self, data):
        # Define dictionaries
        A = {}
        B = {}

        ################## A ##########################
        # Select 18 samples from each of the two classes with outlier=0 from A
        A_outlier_0 = data[(data['WSI'].str.startswith('A')) & (data['outlier'] == 0)].groupby('GT').apply(lambda x: x.sample(n=18, random_state=42))
        # Select 9 samples from each of the 4 classes with outlier=1 from A
        A_outlier_1 = data[(data['WSI'].str.startswith('A')) & (data['outlier'] == 1)].groupby('GT').apply(lambda x: x.sample(n=9, random_state=42))
        
        # Combine the datasets
        A['dataset'] = pd.concat([A_outlier_0, A_outlier_1])
        A['values'] = A['dataset'][A['dataset']['WSI'].str.startswith('A')]['WSI'].values
        A['labels'] = A['dataset'][A['dataset']['WSI'].str.startswith('A')]['outlier'].values
        A['classes'] = A['dataset'][A['dataset']['WSI'].str.startswith('A')]['GT'].values

        ################## B ##########################
        # Select 28 samples from each of the 2 classes with outlier=0 from B
        B_outlier_0 = data[(data['WSI'].str.startswith('B')) & (data['outlier'] == 0)].groupby('GT').apply(lambda x: x.sample(n=28, random_state=42))
        # Select 14 samples from each of the 4 classes with outlier=1 from B
        B_outlier_1 = data[(data['WSI'].str.startswith('B')) & (data['outlier'] == 1)].groupby('GT').apply(lambda x: x.sample(n=14, random_state=42))
    
        # Combine the datasets
        B['dataset'] = pd.concat([B_outlier_0, B_outlier_1])
        B['values'] = B['dataset'][B['dataset']['WSI'].str.startswith('B')]['WSI'].values
        B['labels'] = B['dataset'][B['dataset']['WSI'].str.startswith('B')]['outlier'].values
        B['classes'] = B['dataset'][B['dataset']['WSI'].str.startswith('B')]['GT'].values

        # Print
        print_nbs(A_outlier_1, A_outlier_0, B_outlier_1, B_outlier_0)
        print('B distribution: ', collections.Counter(B['classes']))
        print('A distribution: ', collections.Counter(A['classes']))

        return A, B
    
    def datatrain(self, data, test, hosp, unsupervised):
        dict_data = {}
        test = test['dataset']
        train = data[~data['WSI'].isin(test['WSI'])]
        values_train = train[train['WSI'].str.startswith(hosp)]['WSI'].values
        labels_train = train[train['WSI'].str.startswith(hosp)]['outlier'].values
        if unsupervised:
            dict_data['values'] = values_train[labels_train==0]
            dict_data['labels'] = labels_train[labels_train==0]
        else:
            dict_data['values'] = values_train
            dict_data['labels'] = labels_train
        
        return dict_data
