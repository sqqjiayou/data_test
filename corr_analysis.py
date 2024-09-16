import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os
import itertools
from matplotlib.dates import DateFormatter
import warnings
import time
warnings.filterwarnings('ignore')

class ForexSentimentAnalysis:
    def __init__(self, df1, df2, tag, debug_mode=False):
        """
        Initialize the ForexSentimentAnalysis class.
        
        :param df1: DataFrame with sentiment scores
        :param df2: DataFrame with forex prices
        :param tag: Forex pair tag
        """
        self.df1 = df1.copy()
        self.df2 = df2.copy()
        self.tag = tag
        self.debug_mode = debug_mode
        self.preprocess_data()
        # Create results/records directory if it doesn't exist
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('records'):
            os.makedirs('records')

    def preprocess_data(self):
        """Preprocess the input dataframes."""
        # Convert datetime columns
        self.df1['createDate'] = pd.to_datetime(self.df1['createDate'])
        self.df2['asoftime'] = pd.to_datetime(self.df2['asoftime'], infer_datetime_format=True)

        # Process df2
        self.df2['mid'] = (self.df2['bid'] + self.df2['ask']) / 2
        if self.tag.startswith('USD'):
            for col in ['bid', 'ask', 'mid']:
                self.df2[col] = 1 / self.df2[col]
            self.df2['bid'], self.df2['ask'] = self.df2['ask'], self.df2['bid']

    def resample_data(self, RT, model, window, keyword=''):
        """
        Resample data based on given parameters.
        
        :param RT: Resample time period
        :param model: Model name
        :param window: EMA window size
        :param keyword: Keyword for filtering df1
        :return: Resampled and merged DataFrame
        """
        # Filter df1 by keyword if provided
        if keyword:
            df1_filtered = self.df1[self.df1['keyword'] == keyword]
        else:
            df1_filtered = self.df1

        # Resample df1
        df11 = df1_filtered.resample(RT, on='createDate').agg({
            f"{model}_sentiment_title": 'sum'
        }).reset_index()
        df11['time1'] = df11['createDate'].dt.floor('min')
        ema_window = len(df11)//window
        df11[f"{model}_sentiment_title_{window}"] = df11[f"{model}_sentiment_title"].ewm(span=ema_window, min_periods=len(df11)//5).mean()

        # Resample df2
        df21 = self.df2.resample(RT, on='asoftime').last().reset_index()
        df21['time2'] = df21['asoftime'].dt.floor('min')

        # Merge df11 and df21
        df0 = pd.merge(df11, df21, left_on='time1', right_on='time2', how='outer')
        df0['time'] = df0['time1'].combine_first(df0['time2'])
        df0 = df0.sort_values('time')

        return df0

    def calculate_returns(self, df0, T):
        """
        Calculate returns for different periods.
        
        :param df0: Input DataFrame
        :param T: List of periods for return calculation
        :return: DataFrame with calculated returns
        """
        for t in T:
            df0[f'R_{t}'] = df0['mid'].pct_change(t)
        return df0

    def calculate_correlation(self, df0, model, window, T):
        """
        Calculate correlation between sentiment scores and returns.
        
        :param df0: Input DataFrame
        :param model: Model name
        :param window: EMA window size
        :param T: List of periods for return calculation
        :return: Dictionary of correlations
        """
        correlations = {}
        for t in T:
            sentiment_col = f"{model}_sentiment_title_{window}"
            return_col = f'R_{t}'
            
            # Drop rows where either column has NaN values
            valid_data = df0[[sentiment_col, return_col]].dropna()
            
            if len(valid_data) > 1:  # Ensure there's enough data to calculate correlation
                corr, _ = stats.pearsonr(valid_data[sentiment_col], valid_data[return_col])
                correlations[t] = corr
            else:
                correlations[t] = np.nan
        return correlations

    def process_combination(self, params):
        """
        Process a single parameter combination.
        
        :param params: Tuple of (RT, model, window, keyword, T_list, train_end_date)
        :return: List of results for this combination
        """
        RT, model, window, keyword, T_list, train_end_date = params
        df0_file = f"records/{self.tag}_{RT}_{model}_{window}_{keyword}.csv"

        if os.path.exists(df0_file):
            df0 = pd.read_csv(df0_file, parse_dates=['time'])
        else:
            df0 = self.resample_data(RT, model, window, keyword)
            df0.to_csv(df0_file, index=False)

        df0 = self.calculate_returns(df0, T_list)

        train_df0 = df0[df0['time'] <= train_end_date]
        test_df0 = df0[df0['time'] > train_end_date]

        train_corr = self.calculate_correlation(train_df0, model, window, T_list)
        test_corr = self.calculate_correlation(test_df0, model, window, T_list)

        return [(RT, model, window, keyword, t, train_corr[t], test_corr[t]) for t in T_list]

    def optimize_parameters(self, RT_list, model_list, T_list, window_list, keyword_list, train_end_date, n=None):
        """
        Optimize parameters using grid search and parallel or serial processing.
        
        :param RT_list: List of resample time periods
        :param model_list: List of model names
        :param T_list: List of periods for return calculation
        :param window_list: List of EMA window sizes
        :param keyword_list: List of keywords for filtering
        :param train_end_date: End date for the training set
        :param n: Number of top results to display (optional)
        :return: DataFrame with optimization results
        """
        results = []

        param_combinations = list(itertools.product(RT_list, model_list, window_list, keyword_list))
        param_combinations = [(RT, model, window, keyword, T_list, train_end_date) for RT, model, window, keyword in param_combinations]

        if self.debug_mode:
            for params in param_combinations:
                result = self.process_combination(params)
                results.extend(result)
        else:
            with ProcessPoolExecutor() as executor:
                for result in executor.map(self.process_combination, param_combinations):
                    results.extend(result)

        results_df = pd.DataFrame(results, columns=['RT', 'model', 'window', 'keyword', 'T', 'train_corr', 'test_corr'])
        results_df = results_df.sort_values('train_corr', key=abs, ascending=False)

        # Save all results to Excel
        results_df.to_excel(f'results/{self.tag}_all_results.xlsx', index=False)

        if n:
            results_df = results_df.head(n).reset_index(drop=True)

        return results_df

    def plot_top_correlations(self, results_df, n=5, train_end_date=None):
        """
        Plot top n correlations and time series for the best configuration.
        
        :param results_df: DataFrame with optimization results
        :param n: Number of top results to plot
        :param train_end_date: End date for the training set
        """
        top_n = results_df.head(n)
        
        # Plot correlations
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = range(len(top_n))
        labels = [f"{row.RT}_{row.model}_{row.window}_{row['T']}" for _, row in top_n.iterrows()]
        
        width = 0.35
        ax.bar([i - width/2 for i in x], top_n['train_corr'], width, label='Train', alpha=0.8)
        ax.bar([i + width/2 for i in x], top_n['test_corr'], width, label='Test', alpha=0.8)
        
        ax.set_title(f'Top {n} Correlations (Train vs Test)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Correlation')
        ax.legend()
        
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'results/{self.tag}_top_correlations.png')
        plt.close()

        # Plot time series for the best configuration
        best_config = top_n.iloc[0]
        RT, model, window, keyword, T = best_config[['RT', 'model', 'window', 'keyword', 'T']]
        
        df0_file = f"records/{self.tag}_{RT}_{model}_{window}_{keyword}.csv"
        if os.path.exists(df0_file):
            df0 = pd.read_csv(df0_file, parse_dates=['time'])
        else:
            df0 = self.resample_data(RT, model, window, keyword)
            df0 = self.calculate_returns(df0, [T])
        
        sentiment_col = f"{model}_sentiment_title_{window}"
        
        df_plot = df0.dropna(subset=[sentiment_col, 'mid'])
        
        fig, ax1 = plt.subplots(figsize=(15, 7))
        
        ax2 = ax1.twinx()
        
        train_mask = df_plot['time'] <= train_end_date
        ax1.plot(df_plot[train_mask]['time'], df_plot[train_mask][sentiment_col], color='blue', label=f'{sentiment_col} (Train)')
        ax1.plot(df_plot[~train_mask]['time'], df_plot[~train_mask][sentiment_col], color='blue', linestyle='--', label=f'{sentiment_col} (Test)')
        
        ax2.plot(df_plot[train_mask]['time'], df_plot[train_mask]['mid'], color='red', label='Mid Price (Train)')
        ax2.plot(df_plot[~train_mask]['time'], df_plot[~train_mask]['mid'], color='red', linestyle='--', label='Mid Price (Test)')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel(sentiment_col, color='blue')
        ax2.set_ylabel('Mid Price', color='red')
        
        y1_min, y1_max = df_plot[sentiment_col].min(), df_plot[sentiment_col].max()
        y2_min, y2_max = df_plot['mid'].min(), df_plot['mid'].max()
        
        y1_range = y1_max - y1_min
        y2_range = y2_max - y2_min
        
        ax1.set_ylim(y1_min - 0.1*y1_range, y1_max + 0.1*y1_range)
        ax2.set_ylim(y2_min - 0.1*y2_range, y2_max + 0.1*y2_range)
        
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Calculate correlations for train and test sets
        train_corr = df_plot[train_mask][[sentiment_col, 'mid']].corr().iloc[0, 1]
        test_corr = df_plot[~train_mask][[sentiment_col, 'mid']].corr().iloc[0, 1]
        
        plt.title(f'Sentiment vs Mid Price for {self.tag}\n{sentiment_col}, RT={RT}, T={T}\n'
                  f'Train Corr: {train_corr:.4f}, Test Corr: {test_corr:.4f}')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        plt.show()
        # plt.savefig(f'results/{self.tag}_best_config_time_series.png')
        # plt.close()

# Test example
if __name__ == "__main__":
    start = time.time()
    tag = 'USDCNH'
    tag_short = tag.replace('USD','')
    debug = True
    df1 = pd.read_csv(f'data/{tag}_news_scores.csv')
    df1 = df1.rename(columns={f'ABSA_Bert_sentiment_title_{tag_short}':'ABSA_sentiment_title'})

    df2 = pd.read_csv(f'data/{tag}.csv')

    # Initialize the ForexSentimentAnalysis class with debug_mode=True for debugging
    fsa = ForexSentimentAnalysis(df1, df2, tag=tag, debug_mode=debug)

    # Define parameters for optimization
    RT_list = ['15min', '10min', '30min']
    model_list = ['Vader', 'ABSA', 'FinBERT']
    T_list = [1, 5, 10, 15, 30, 60, 120]
    window_list = [20]
    keyword_list = ['']
    train_end_date = '2024-04-30'

    # Run optimization
    results = fsa.optimize_parameters(RT_list, model_list, T_list, window_list, keyword_list, train_end_date, n=10)

    # Print results
    print(results)

    # Plot top correlations and time series
    fsa.plot_top_correlations(results, n=10, train_end_date=pd.to_datetime(train_end_date))
    print(f'Total running time: {time.time()-start} seconds.')