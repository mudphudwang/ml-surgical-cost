import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from collections import OrderedDict
from distutils.dir_util import copy_tree

def format_df(df_raw):
    df_data = {}
    df_data['Gender'] = df_raw['Gender'].values
    df_data['Race'] = df_raw['Race'].values
    df_data['Ethnicity'] = df_raw['Ethnicity'].values
    df_data['Age Group'] = df_raw['Age Group'].values
    df_data['Type of Admission'] = df_raw['Type of Admission'].values
    df_data['APR Severity of Illness Code'] = df_raw['APR Severity of Illness Code'].values
    df_data['APR Risk of Mortality'] = df_raw['APR Risk of Mortality'].values
    df_data['Payment Typology 1'] = df_raw['Payment Typology 1'].values
    df_data['Discharge Year'] = df_raw['Discharge Year'].values
    df_data['CCS Diagnosis Code'] = df_raw['CCS Diagnosis Code'].fillna(0).values.astype(np.int)
    df_data['Patient Disposition'] = df_raw['Patient Disposition'].values
    remove_comma = np.vectorize(lambda x: x.replace(',', ''))
    df_data['Total Costs'] = remove_comma(df_raw['Total Costs'].values).astype(np.float)
    remove_plus = np.vectorize(lambda x: x.replace(' +', ''))
    df_data['Length of Stay'] = remove_plus(df_raw['Length of Stay'].values).astype(np.int)
    return pd.DataFrame(df_data)

def process_df(df, dispo_cutoff, cost_cutoff=0.5, los_split=(2,5)):
    df_years = []
    print('Dispositions Excluded:')
    for year in range(2009, 2017):
        print('  {}:'.format(year))
        df_year = df[df['Discharge Year']==year]

        # Filter out the tail percentiles
        year_costs = df_year['Total Costs'].values
        lower_cost = np.percentile(year_costs, cost_cutoff)
        upper_cost = np.percentile(year_costs, 100-cost_cutoff)
        df_year = df_year[df_year['Total Costs'] > lower_cost]
        df_year = df_year[df_year['Total Costs'] < upper_cost]

        # Filter out null dispositions
        null_dispo = df_year['Patient Disposition'].isnull()
        if sum(null_dispo) > 0:
            print('    Null: {}'.format(sum(null_dispo)))
            df_year = df_year[~null_dispo]

        # Filter out select dispositions
        for _dispo_cutoff in dispo_cutoff:
            bad_dispo = df_year['Patient Disposition'] == _dispo_cutoff
            if sum(bad_dispo) > 0:
                print('    {}: {}'.format(_dispo_cutoff, sum(bad_dispo)))
            df_year = df_year[~bad_dispo]

        assert not df_year.isnull().any().any()

        df_years.append(df_year)
    df_filt = pd.concat(df_years).reset_index(drop=True)
    return df_filt

def fit_encoders(df):
    input_cats = [
        'APR Risk of Mortality',
        'APR Severity of Illness Code',
        'Age Group',
        'CCS Diagnosis Code',
        'Ethnicity',
        'Gender',
        'Payment Typology 1',
        'Race',
        'Type of Admission']
    output_cats = [
        'Total Costs Class',
        'Length of Stay Class']
    input_cats.sort()
    output_cats.sort()
    encoders_input = OrderedDict()
    for input_cat in input_cats:
        lb = preprocessing.LabelBinarizer()
        encoders_input[input_cat] = lb.fit(df[input_cat].values)
    return encoders_input

def split_train_test(df, split_index):
    assert split_index in range(0,5), 'split_index must be between 0 and 4'
    test_years = [year for year in range(2012, 2017)]
    test_year = test_years[split_index]
    train_years = [year for year in range(test_year-3, test_year)]
    df_train = df[df['Discharge Year'] >= train_years[0]]
    df_train = df_train[df_train['Discharge Year'] <= train_years[-1]].reset_index(drop=True)
    df_test = df[df['Discharge Year'] == test_year].reset_index(drop=True)
    return df_train, df_test

def encode_data(df, encoders_input):
    inp_list = []
    for n,v in encoders_input.items():
        inp_list.append(v.transform(df[n].values))
    inp = np.concatenate(inp_list, axis=1)
    return inp

class Data():
    """
    Parameters:
        apr_drg_code: int
            APR DRG Code
        data_dir: str
            directory containing csv file
        output_type: str
            {'los', 'cost'}
        dispo_cutoff: list/tuple
            Patient Dispositions to discard
        cost_cutoff: float
            Tail percentile cutoffs
        split_train_val: bool
            Split train dataset into train (90%) and validation (10%)
    Attributes:
        apr_drg_code: int
        df: pd.DataFrame
            Dataframe containing
        encoders: list of binary encoders
        datasets: list of dictionaries containing train/test datasets
    """
    def __init__(self, apr_drg_code, data_dir, output_type, dispo_cutoff,
        cost_cutoff=0.5, split_train_val=False):
        assert output_type in ['los', 'cost']
        if output_type == 'los':
            out_str = 'Length of Stay'
        elif output_type == 'cost':
            out_str = 'Total Costs'
        fname = os.path.join('/root/local/', data_dir, '{}.csv'.format(apr_drg_code))
        if not os.path.exists(fname):
            print('transferring data from mount to local ...')
            mnt_dir = os.path.join('/root/data/', data_dir)
            local_dir = os.path.join('/root/local/', data_dir)
            copy_tree(mnt_dir, local_dir)
            print('done transfering')
        df_all = format_df(pd.read_csv(fname, low_memory=False))
        self.output_type = output_type
        self.apr_drg_code = apr_drg_code
        self.df = process_df(format_df(pd.read_csv(fname, low_memory=False)),
               dispo_cutoff, cost_cutoff)
        self.df = self.df[self.df['Payment Typology 1'] == 'Medicare']
        self.encoders_input = fit_encoders(self.df)
        self.datasets = []
        for dset_ind in range(0,5):
            dfs = {}
            dfs['train'], dfs['test'] = split_train_test(self.df, dset_ind)
            self.datasets.append({})
            for name, df in dfs.items():
                dset = {}
                # dset['year'] = df['Discharge Year'].unique()
                dset['year'] = df['Discharge Year'].values
                dset['input'] = encode_data(df, self.encoders_input)
                dset['output'] = df[out_str].values
                self.datasets[-1][name] = dset
            if split_train_val:
                len_all = self.datasets[-1]['train']['input'].shape[0]
                len_train = int(len_all * 0.9)
                train_ind = np.random.choice(
                    np.arange(len_all), size=len_train, replace=False)
                train_mask = np.isin(np.arange(len_all), train_ind)
                val_mask = np.logical_not(train_mask)
                self.datasets[-1]['val'] = {'year':self.datasets[-1]['train']['year']}
                for n,v in self.datasets[-1]['train'].items():
                    if n == 'year': continue
                    self.datasets[-1]['train'][n] = v[train_mask]
                    self.datasets[-1]['val'][n] = v[val_mask]
        normalize_output(self.datasets)

    def __repr__(self):
        return 'Data from 2009 to 2016 for APR DRG Code: {}'.format(self.apr_drg_code)

def normalize_output(datasets):
    for dataset in datasets:
        for k in dataset.keys():
            output_n = dataset[k]['output'].copy()
            output_means = np.zeros_like(output_n)
            output_stdevs = np.zeros_like(output_n)
            for y in np.unique(dataset[k]['year']):
                year_mask = dataset[k]['year'] == y
                output_mean = output_n[year_mask].mean()
                output_std = output_n[year_mask].std()
                output_n[year_mask] = ((output_n - output_mean) / output_std)[year_mask]
                output_means[year_mask] = output_mean
                output_stdevs[year_mask] = output_std
            dataset[k]['output_normalized'] = output_n
            dataset[k]['output_mean'] = output_means
            dataset[k]['output_stdev'] = output_stdevs
