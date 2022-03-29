import datetime
import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import six
from sklearn.cluster import KMeans
from statsmodels.sandbox.tools.tools_pca import pca

import MacPy.datafetcher.datafetcher as mdf
from MacPy.data.datamart import Datamart
from MacPy.tablemanager import TableManager

import constants as consts
import data as msdata
import utilities as msutils

logger = logging.getLogger(__name__)

# ### ### ### PCA-specific constants ### ### ### #
PCA_VAR_CUTOFF = 0.80  # in percentage
PCA_LOOKBACK_PERIODS = 180
PCA_DEMEAN = True
KMEANS_NUM_CLUSTERS = 7

PLOT_MARKER_MAP = {
    'fund': 'o',  # circle for y funds
    'cluster': 'x',  # cross for Cluster Center
    'factor': 'd',  # square
    'americas': 's',  # square
    'eu': 'd',  # diamond
    'asia': 'v',  # triangle - down
    'em': '^',  # triangle - up
    'global': '*',  # star
    'pentagon': 'p',  # unused yet
    'default': 'h'  # hexagon
}  # marker by regions

PLOT_COLOR_MAP = {
    'fund': 'maroon',
    'cluster': 'black',
    'fx': 'purple',
    'bond': 'steelblue',
    'credit': 'lightblue',
    'equity': 'green',
    'commodity': 'orange',
    'default': 'yellow'
}  # plot color by asset type


# TODO: a bug exists when factors=None, defaults, self.x is not loaded, but self.xdata exists


class MorningstarPCA(object):
    # Class Variables
    tm = None
    dm = None

    def __init__(self, factors=None, funds=None, demean=PCA_DEMEAN, **kwargs):
        if self.tm is None:
            self.tm = TableManager()
        if self.dm is None:
            self.dm = Datamart()

        # Factors, to determine the principal components PC's
        self.xdata = None  # type of FactorData
        self._set_xdata(factors=factors, **kwargs)

        # Fund data, whose driving factors are to be determined
        self.ydata = None
        self.set_ydata(funds, **kwargs)

        # PCA related variables
        self.x = None  # type of DataFrame
        self.demean = demean
        self.xreduced = None
        self.pcs = None
        self.evals = None
        self.evecs = None
        self.evecs_inv = None
        self.var_explained = None
        self.var_cumsum = None
        self.pc_num = None
        self.pc_syms = None

        # Coordinates of factor (x), funds (y), cluster center (c) onto PC plain
        self.x_coords = None
        self.y_coords = None
        self.c_coords = None

        # KMean Cluster
        self.pca_cluster = None  # pd.Series: factor sym -> cluster number (int)
        self.center = None  # pd.Series: factor -> cluster center C0, C1, etc. (string)
        self.center2factor = None  # pd.Series: cluster center C_i to identifying factor e.g. USDCAD

        self.cluster_dist = {}  # dict: {C_i -> pd.Series(dist to C_i, index=factors in cluster C_i)}
        self.cluster_members = {}  # dict: {C_i -> list[factors belong to cluster C_i], sorted}

        # Fund Y-Data
        self.y = None
        self.fund2center = None

    def _set_xdata(self, factors=None, **kwargs):
        if factors is None:
            df_factors = self.tm.load_table(consts.MSTAR_PCA_FACTORS)
            factors = df_factors['ticker'].unique().tolist()
            self.xdata = msdata.FactorData(ids=factors, **kwargs)
        elif isinstance(factors, msdata.BaseData):
            self.xdata = factors
        else:
            self.xdata = msdata.FactorData(ids=factors, **kwargs)
        self.xdata.load_data()

    def set_ydata(self, funds=None, **kwargs):
        if funds is None:
            return  # no necessary to setup self.ydata to run PCA
        elif isinstance(funds, msdata.BaseData):  # FactorData, FundData are all BaseData
            self.ydata = funds
        elif isinstance(funds, list):  # list of tickers
            self.ydata = msdata.FactorData(ids=funds, **kwargs)
        else:
            raise TypeError('Invalid input type: type(funds) = {}'.format(type(funds)))
        self.ydata.load_data()

    def fit(self):
        self.fit_factors()
        if self.ydata is None:
            logger.warning('Fund data (ydata) are not specified in PCA Model. ')
            return
        self.map_funds()

    def fit_factors(self):
        self.x = self.xdata.returns.dropna()  # self.xdata -> self.x
        if self.x.empty:
            raise ValueError('self.xdata.returns.dropna() is empty, please double check its values. ')
        self.run_pca()  # self.x -> self.pcs
        # Projecting:  (self.x, self.pcs) -> self.x_coords
        self.x_coords = self.pc_project(self.x, self.pcs)[self.pc_syms]

        self.run_kmean()  # self.x_coords -> self.c_coords, self.pca_cluster
        # Find the nearest cluster center to each factor, should coincide with self.cluster_members
        self.center = self.pc_cluster(self.x_coords, self.c_coords)
        self.identify_clusters()  # -> self.center2factor

    def map_funds(self):
        self.y = self.ydata.returns.ix[self.pcs.index].ffill()  # or dropna()
        # Projecting: (self.y, self.pcs) -> self.y_coords
        self.y_coords = self.pc_project(self.y, self.pcs)[self.pc_syms]
        # Clustering:  # self.y_coords -> self.fund2center
        self.fund2center = self.pc_cluster(self.y_coords, self.c_coords)
        return self.driving_factors

    def run_pca(self, demean=None, pc_num=0):
        if self.x is None or (isinstance(self.x, pd.DataFrame) and self.x.empty):
            raise ValueError('No x data, please re-instantiate with input factors.')
        pc_syms = ['PC{}'.format(x) for x in range(len(self.x.columns))]

        demean = self.demean if demean is None else demean
        xreduced, pcs, evals, evecs = pca(self.x.values, demean=demean)
        self.xreduced = pd.DataFrame(xreduced, columns=self.x.columns, index=self.x.index)
        self.pcs = pd.DataFrame(pcs, columns=pc_syms, index=self.x.index)
        self.evecs = pd.DataFrame(evecs, columns=pc_syms, index=self.x.columns)
        self.evecs_inv = pd.DataFrame(np.linalg.inv(evecs), columns=self.x.columns, index=pc_syms)
        self.evals = pd.Series(evals, index=pc_syms)

        self.var_explained = self.evals / self.evals.sum()
        self.var_cumsum = self.var_explained.cumsum()
        if isinstance(pc_num, six.integer_types) and pc_num > 0:
            self.pc_num = min(pc_num, len(self.x.columns))
        else:
            self.pc_num = max(2, np.argmax(self.var_cumsum.values > PCA_VAR_CUTOFF))  # Integer 2, not symbol PC2
        self.pc_syms = pc_syms[:self.pc_num]

    def run_kmean(self, n_cluster=KMEANS_NUM_CLUSTERS):
        # Find Clusters by K-Mean
        if isinstance(n_cluster, six.integer_types) and n_cluster > 0:
            n_cluster = min(n_cluster, len(self.x.columns))  # cannot be more than number of factors
        else:
            n_cluster = int(round(math.sqrt(0.5 * len(self.x.columns))))
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(self.x_coords.values)

        # self.x2cluster
        # Series: map Factor -> cluster integer, e.g. 'USDCAD' -> 2
        self.pca_cluster = pd.Series(kmeans.labels_, index=self.x_coords.index, name='cluster')
        # pd.concat([self.x_coords, self.cluster], axis=1)  # DataFrame with cluster int labels

        # Cluster center coordinates, index = [C0, C1, etc.], columns = [PC0, PC1, etc.]
        c_coords = pd.DataFrame(kmeans.cluster_centers_)
        c_coords.columns = self.x_coords.columns
        c_coords.index = ['C{}'.format(x) for x in sorted(list(set(kmeans.labels_)))]
        self.c_coords = c_coords

    def identify_clusters(self):  # by given x-factors
        # Map from cluster center to the nearest factor
        df_dist = self.pc_distance(self.x_coords, self.c_coords).T  # note: .T here
        self.center2factor = self.get_column_with_min_value(df_dist)

        # Center i to the list of factors belong to cluster C_i
        c2factor_list = {c: [] for c in self.center.unique()}
        for k, v in self.center.iteritems():
            c2factor_list[v].append(k)

        cluster_dist = dict.fromkeys(self.center.unique())
        for c, row in df_dist.iterrows():
            cluster_dist[c] = row[row.index.isin(c2factor_list.get(c, []))].sort_values()  # pd.Series
        self.cluster_dist = cluster_dist  # {k: v is pd.Series(factor->distance to k)}
        self.cluster_members = {k: v.index.tolist() for k, v in cluster_dist.items()}

    @property
    def driving_factors(self):
        if self.center2factor is None or len(self.center2factor) == 0:  # type of pd.Series
            logger.warning('Cluster centers are not identified, call fit_factors()')
            self.fit_factors()
        if self.fund2center is None or len(self.fund2center) == 0:  # type of pd.Series
            logger.warning('Funds are not mapped to principal components, call map_funds()')
            self.map_funds()
        return {k: self.center2factor.get(v) for k, v in self.fund2center.to_dict().items()}

    def rolling_driving_factors(self, start_date=None, end_date=None, periods=None):
        if start_date is None:
            # start = max(self.xdata.loading_date, self.ydata.loading_date)
            start = max(x for x in [self.xdata.loading_date, self.ydata.loading_date] if x is not None)
        else:
            start = pd.to_datetime(start_date)
            if start > self.xdata.load_date:
                self.xdata.loading_date = start
                self.xdata.load_data()
            if start > self.ydata.loading_date:
                self.ydata.loading_date = start
                self.ydata.load_data()
        end = pd.to_datetime(datetime.date.today()) if end_date is None else pd.to_datetime(end_date)
        periods = self.xdata.periods if periods is None else periods

        # fixme: use downsample instead
        month_end_range = msutils.get_month_end_range(start, end)
        dict_drivers = dict.fromkeys(month_end_range)
        for month_end in month_end_range:
            self.xdata.set_dates(end_date=month_end, periods=periods)
            self.ydata.set_dates(date_index=self.xdata.date_index)
            self.xdata.recompute_data()
            self.ydata.recompute_data()
            self.fit()
            dict_drivers[month_end] = self.driving_factors
        df_driver = pd.DataFrame(dict_drivers).T
        df_driver.index.name = 'Date'
        return df_driver

    # ### ### ### ### ### ### Static Methods ### ### ### ### ### ### #
    # Maybe move to MacPy.morningstar.utilities as msutils
    @staticmethod
    def pc_project(data, pcs):
        """Project data onto principal components (pcs)"""
        if not isinstance(pcs, pd.DataFrame):
            TypeError('Invalid input type: type(pcs) = {}'.format(type(pcs)))
        if isinstance(data, pd.DataFrame):
            res = {sym: {pc: data[sym].dot(pcs[pc]) for pc in pcs} for sym in data}
            return pd.DataFrame(res).T
        elif isinstance(data, pd.Series):
            res = {pc: data.dot(pcs[pc]) for pc in pcs}
            return pd.Series(res, name=data.name)
        else:
            TypeError('Invalid input type: type(data) = {}'.format(type(data)))

    @staticmethod
    def pc_cluster(data, clusters):
        """Group data into clusters depending on the distance. """
        dist = MorningstarPCA.pc_distance(data, clusters)
        return MorningstarPCA.get_column_with_min_value(dist)

    @staticmethod
    def pc_distance(data, target):
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Invalid input type: type(data) = {}'.format(type(data)))
        if not isinstance(target, pd.DataFrame):
            raise TypeError('Invalid input type: type(target) = {}'.format(type(target)))
        if not all(data.columns == target.columns):
            error_str = 'data.columns = {}'.format(', '.join(data.columns))
            error_str += ', target.columns = {}'.format(', '.join(target.columns))
            ValueError('Columns in data and target do not match: {}'.format(error_str))
        dist = dict.fromkeys(target.index)
        for center, row in target.iterrows():
            dist[center] = ((data - row) ** 2).sum(axis=1).apply(np.sqrt)
        return pd.DataFrame(dist)

    @staticmethod
    def get_column_with_min_value(data):
        """Return the column name with minimum value for each index value"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Invalid input type: type(data) = {}'.format(type(data)))
        min_col_name = pd.Series(index=data.index)
        for idx, row in data.iterrows():
            min_col_name[idx] = row.argmin()
        return min_col_name

    @staticmethod
    def get_sorted_columns(data):
        """Return the column names sorted by their values each index"""
        if not isinstance(data, pd.DataFrame):
            raise TypeError('Invalid input type: type(data) = {}'.format(type(data)))
        col_names = pd.Series(index=data.index)
        for idx, row in data.iterrows():
            col_names[idx] = row.sort_values().index.tolist()
        return col_names

    # ### ### ### ### ### ### Plot Functions ### ### ### ### ### ### #
    # self.xdata.values.ffill().dropna().plot(figsize=(15, 10))

    def plot_variance(self, **kwargs):
        df_var = pd.DataFrame({'Explained': self.var_explained, 'Cumulative': self.var_cumsum})
        df_var.rename(index={x: x.replace('PC', '') for x in df_var.index}, inplace=True)
        return df_var.plot(kind='bar', figsize=(15, 5), title='Variance Percentage', **kwargs)

    def plot_evecs(self, n=None, **kwargs):
        if n is None:
            n = self.pc_num
        elif not isinstance(n, six.integer_types):
            raise TypeError('Invalid input type, type(pc_num) = {}'.format(type(n)))
        if self.evecs is None:
            logger.warning('PCA Eigenvectors are not yet implemented.')
            return
        if n > len(self.evecs.columns):
            n = len(self.evecs.columns)
        pcs = ['PC{}'.format(x) for x in range(n)]
        pcs = [col for col in pcs if col in self.evecs.columns]
        return self.evecs[pcs].plot(kind='bar', subplots=True, figsize=(15, 5 * n),
                                    title='Principal Component Coefficients', **kwargs)

    # Scatter features:   color   |   marker    | alpha | size
    # x_factors:       5+1 assets | 4+1 regions |  0.5  | 100
    # y_funds (target):    red    |   circle 0  |  0.7  | 150
    # c_cluster center:   black   |   cross x   |  0.2  | 200-300
    def plot_coords(self, show_text=True, **kwargs):  # xlim = [-0.05, 0.09], ylim=[-0.005, 0.005]
        pc0 = self.pc_syms[0]  # PC0
        pc1 = self.pc_syms[1]  # PC1

        dfx = self.x_coords
        dfx_info = self.tm.load_table(consts.MSTAR_PCA_FACTORS)
        dfx_info.index = dfx_info['ticker'].map(lambda x: x.split(' ')[0])  # to be pre-computed
        dfx_info.index.name = 'sym'
        dfx = dfx.merge(dfx_info, how='left', left_index=True, right_index=True)
        default_marker = PLOT_MARKER_MAP.get('default')
        default_color = PLOT_COLOR_MAP.get('default')
        dfx['marker'] = dfx['region'].map(lambda x: PLOT_MARKER_MAP.get(x, default_marker))
        dfx['color'] = dfx['asset'].map(lambda x: PLOT_COLOR_MAP.get(x, default_color))
        dfx.drop(dfx_info.columns.tolist(), axis=1, inplace=True, errors='ignore')

        ax = dfx.iloc[0:1, :].plot.scatter(x=pc0, y=pc1, figsize=(10, 10), linewidth=0, s=0, color='w', **kwargs)
        for m in dfx['marker'].unique():
            for c in dfx['color'].unique():
                msg = '({}, {})'.format(m, c)
                df = dfx[(dfx['marker'] == m) & (dfx['color'] == c)]
                if df.empty:
                    logger.debug(msg + ': empty')
                    continue
                else:
                    logger.debug(msg + ': {}'.format(len(df)))
                    df.plot.scatter(x=pc0, y=pc1, ax=ax, color=c, marker=m, s=150, alpha=0.5)

        dfy = self.y_coords
        m = PLOT_MARKER_MAP.get('fund')
        c = PLOT_COLOR_MAP.get('fund')
        dfy.plot.scatter(x=pc0, y=pc1, ax=ax, color=c, marker=m, s=200, alpha=0.8)

        if show_text:
            for name, row in dfx.iterrows():
                ax.annotate(name, xy=[row[pc0], row[pc1]])
            for name, row in dfy.iterrows():
                ax.annotate(name, xy=[row[pc0], row[pc1]], color=c)

        dfc = self.c_coords
        m = PLOT_MARKER_MAP.get('cluster')
        c = PLOT_COLOR_MAP.get('cluster')
        dfc.plot.scatter(x=pc0, y=pc1, ax=ax, color=c, marker=m, s=200, alpha=0.8)

    def plot_clusters(self, show_text=True, **kwargs):
        pc0 = self.pc_syms[0]  # PC0
        pc1 = self.pc_syms[1]  # PC1

        dfx = self.x_coords.merge(pd.DataFrame(self.pca_cluster), how='left', left_index=True, right_index=True)
        m = PLOT_MARKER_MAP.get('factor', 'h')
        cmap = self.get_list_colors(self.pca_cluster.unique())

        k = 0  # there is at least one cluster
        df = dfx[dfx['cluster'] == k]
        ax = df.plot.scatter(x=pc0, y=pc1, figsize=(10, 10), color=cmap.get(k), marker=m, s=150, alpha=0.75, **kwargs)
        for k in sorted(dfx['cluster'].unique()):
            if k == 0:
                continue
            df = dfx[dfx['cluster'] == k]
            df.plot.scatter(x=pc0, y=pc1, ax=ax, color=cmap.get(k), marker=m, s=100, alpha=0.75)

        dfy = self.y_coords
        m = PLOT_MARKER_MAP.get('fund')
        c = PLOT_COLOR_MAP.get('fund')
        dfy.plot.scatter(x=pc0, y=pc1, ax=ax, color=c, marker=m, s=200, alpha=0.8)

        if show_text:
            for name, row in dfx.iterrows():
                ax.annotate(name, xy=[row[pc0], row[pc1]])
            for name, row in dfy.iterrows():
                ax.annotate(name, xy=[row[pc0], row[pc1]], color=c)

        dfc = self.c_coords
        m = PLOT_MARKER_MAP.get('cluster')
        c = PLOT_COLOR_MAP.get('cluster')
        dfc.plot.scatter(x=pc0, y=pc1, ax=ax, color=c, marker=m, s=200, alpha=0.8)

    @staticmethod
    def get_list_colors(in_list, cmap='Set2'):
        if isinstance(in_list, six.integer_types):
            alist = range(in_list)
        else:
            alist = list(in_list)
        cmap = plt.get_cmap(cmap)  # 'Set1', 'Set2', 'Paired'
        cval = np.linspace(0, 1, len(alist))
        return {x: cmap(cval[idx]) for idx, x in enumerate(alist)}


# ### ### ### Miscellaneous Code to generate PCA related reference tables in MongoDB ### ### ### #
# from MacPy.tablemanager import TableManager
# tm = TableManager()

# df_pca_factors = generate_pca_factors_for_mongo()
# tm.save_table(name='mstar_pca_factors', table=df_pca_factors)

# df_factors = tm.load_table('mstar_pca_factors')
# df_factors


def pca_factor_universe_remap(factor_universe, factor_symbol='ticker',
                              outer_key='asset', inner_key='region'):
    records = []
    for out_key, out_val in factor_universe.iteritems():
        for in_key, in_val in out_val.iteritems():
            records.extend([{factor_symbol: x.upper(), inner_key: in_key, outer_key: out_key} for x in in_val])
    return pd.DataFrame(records)[[factor_symbol, outer_key, inner_key]]


def generate_pca_factors_for_mongo(include_name=True):
    df_factors = pd.DataFrame(pca_factor_universe_remap(consts.PCA_FACTOR_UNIVERSE))
    df_factors['id'] = df_factors['ticker'].map(lambda x: x.split()[0])
    df_factors = df_factors.set_index('id').reset_index()  # move column 'id' first

    if include_name:  # names from Bloomberg, need a Bloomberg terminal open
        ticker_list = df_factors['ticker'].tolist()
        field_list = ['name', 'short_name', 'security_name']
        name_dict = mdf.get_ref_data_bbg_wrapper(ticker_list, field_list)
        df_names = pd.DataFrame(name_dict).T.reset_index()
        df_names.rename(columns={'index': 'ticker'}, inplace=True)
        df_factors = df_factors.merge(df_names, on='ticker')
    return df_factors
