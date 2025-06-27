"""
Correlation Analyzer class for Retail Analytics System
"""

import time
import logging
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """
    Correlation analyzer that finds relationships between various customer metrics
    
    This class analyzes relationships between customer demographics, behaviors
    (dwell time, path patterns), and store performance metrics.
    """
    
    def __init__(self, config):
        """
        Initialize the correlation analyzer with the provided configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        
        # Correlation parameters
        self.min_data_points = config.get('min_data_points', 10)
        self.significance_threshold = config.get('significance_threshold', 0.05)
        self.time_window = config.get('time_window', 3600)
        
        # Storage for various metrics
        self.demographic_data = []
        self.dwell_data = []
        self.path_data = []
        self.zone_metrics = defaultdict(list)
        self.temporal_data = []
        self.sales_data = []
        self.has_sales_data = False
        
        # Cache for correlation results
        self.correlation_cache = {}
        self.last_correlation_time = 0
        self.cache_expiry = config.get('cache_expiry', 300)
        
        logger.info("Correlation analyzer initialized")
    
    def _check_cache_and_data(self, cache_key, data_list):
        """
        Helper method to check cache and validate data sufficiency
        
        Args:
            cache_key (str): Cache key for the correlation type
            data_list (list): Data list to validate
            
        Returns:
            dict or None: Cached result if available, None if needs processing
        """
        # Check cache
        if (cache_key in self.correlation_cache and 
            time.time() - self.last_correlation_time < self.cache_expiry):
            return self.correlation_cache[cache_key]
        
        # Check data sufficiency
        if len(data_list) < self.min_data_points:
            result = {
                'status': 'insufficient_data',
                'min_required': self.min_data_points,
                'data_count': len(data_list)
            }
            self._cache_result(cache_key, result)
            return result
        
        return None
    
    def _cache_result(self, cache_key, result):
        """Cache correlation result"""
        self.correlation_cache[cache_key] = result
        self.last_correlation_time = time.time()
    
    def _calculate_pearson_correlation(self, x, y, min_samples=5):
        """
        Calculate Pearson correlation with significance testing
        
        Args:
            x, y: Data arrays for correlation
            min_samples: Minimum samples required
            
        Returns:
            dict: Correlation results with significance
        """
        if len(x) < min_samples or len(y) < min_samples:
            return None
        
        # Ensure equal length
        min_len = min(len(x), len(y))
        x_trimmed = x[:min_len]
        y_trimmed = y[:min_len]
        
        corr, p_value = stats.pearsonr(x_trimmed, y_trimmed)
        
        return {
            'pearson_r': corr,
            'p_value': p_value,
            'significant': p_value < self.significance_threshold,
            'samples': min_len
        }
    
    def _limit_data_size(self, data_list, max_entries_key, default_max=1000):
        """Limit data list size to prevent memory issues"""
        max_entries = self.config.get(max_entries_key, default_max)
        if len(data_list) > max_entries:
            return data_list[-max_entries:]
        return data_list
    
    def add_demographic_data(self, demographics):
        """Add demographic data for correlation analysis"""
        for person_id, demo in demographics.items():
            if all(k in demo for k in ('age', 'gender', 'race', 'emotion')):
                data_point = {
                    'person_id': person_id,
                    'timestamp': time.time(),
                    'age': demo.get('age', 0),
                    'gender': demo.get('gender', 'Unknown'),
                    'race': demo.get('race', 'Unknown'),
                    'emotion': demo.get('emotion', 'Unknown'),
                    'confidence': demo.get('confidence', 0.0)
                }
                self.demographic_data.append(data_point)
        
        self.demographic_data = self._limit_data_size(
            self.demographic_data, 'max_demographic_entries'
        )
    
    def add_dwell_data(self, dwell_stats):
        """Add dwell time data for correlation analysis"""
        timestamp = time.time()
        
        for zone_id, stats in dwell_stats.items():
            data_point = {
                'zone_id': zone_id,
                'timestamp': timestamp,
                'avg_duration': stats.get('avg_duration', 0),
                'total_dwells': stats.get('total_dwells', 0),
                'unique_visitors': stats.get('unique_visitors', 0)
            }
            self.dwell_data.append(data_point)
            self.zone_metrics[zone_id].append(data_point)
        
        self.dwell_data = self._limit_data_size(self.dwell_data, 'max_dwell_entries')
        
        # Limit zone metrics as well
        for zone_id in self.zone_metrics:
            self.zone_metrics[zone_id] = self._limit_data_size(
                self.zone_metrics[zone_id], 'max_dwell_entries'
            )
    
    def add_path_data(self, paths, transitions):
        """Add path data for correlation analysis"""
        timestamp = time.time()
        
        # Process path data
        for person_id, path in paths.items():
            if not path or len(path) < 2:
                continue
                
            start_time = path[0][2]
            end_time = path[-1][2]
            duration = end_time - start_time
            
            if duration < 1.0:  # Skip very short paths
                continue
            
            # Calculate total distance efficiently
            coordinates = np.array([(p[0], p[1]) for p in path])
            total_distance = np.sum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))
            
            data_point = {
                'person_id': person_id,
                'timestamp': timestamp,
                'path_length': len(path),
                'path_distance': total_distance,
                'path_duration': duration,
                'avg_velocity': total_distance / duration if duration > 0 else 0,
                'zones_visited': len(set(p[3] for p in path if p[3] is not None))
            }
            self.path_data.append(data_point)
        
        # Process transition data
        for (source, target), count in transitions.items():
            if source is not None and target is not None:
                self.temporal_data.append({
                    'timestamp': timestamp,
                    'source_zone': source,
                    'target_zone': target,
                    'transition_count': count
                })
        
        self.path_data = self._limit_data_size(self.path_data, 'max_path_entries')
        self.temporal_data = self._limit_data_size(
            self.temporal_data, 'max_temporal_entries', 5000
        )
    
    def add_sales_data(self, sales_data):
        """Add sales data for correlation with customer behavior"""
        if not sales_data:
            return
            
        self.has_sales_data = True
        
        if 'timestamp' not in sales_data:
            sales_data['timestamp'] = time.time()
            
        self.sales_data.append(sales_data)
        self.sales_data = self._limit_data_size(self.sales_data, 'max_sales_entries')
    
    def get_demographic_correlations(self):
        """Calculate correlations between demographic factors"""
        cache_key = 'demographic'
        cached_result = self._check_cache_and_data(cache_key, self.demographic_data)
        if cached_result is not None:
            return cached_result
        
        df_demographics = pd.DataFrame(self.demographic_data)
        results = {'status': 'success'}
        
        # Age distribution analysis
        age_groups = []
        for demo in self.demographic_data:
            age = demo.get('age', 0)
            if age < 18:
                group = '<18'
            elif age < 30:
                group = '18-29'
            elif age < 45:
                group = '30-44'
            elif age < 60:
                group = '45-59'
            else:
                group = '60+'
            age_groups.append({'person_id': demo['person_id'], 'age_group': group})
        
        df_age_groups = pd.DataFrame(age_groups)
        
        # Calculate distributions
        if not df_age_groups.empty and 'age_group' in df_age_groups.columns:
            age_group_stats = df_age_groups.groupby('age_group').size()
            results['age_distribution'] = age_group_stats.to_dict()
        
        for column in ['gender', 'emotion', 'race']:
            if column in df_demographics.columns:
                distribution = df_demographics[column].value_counts().to_dict()
                results[f'{column}_distribution'] = distribution
        
        self._cache_result(cache_key, results)
        return results
    
    def get_dwell_correlations(self):
        """Calculate correlations between dwell time and other metrics"""
        cache_key = 'dwell'
        cached_result = self._check_cache_and_data(cache_key, self.dwell_data)
        if cached_result is not None:
            return cached_result
        
        df_dwell = pd.DataFrame(self.dwell_data)
        results = {'status': 'success', 'zone_metrics': {}, 'overall_stats': {}}
        
        # Calculate overall statistics
        if 'avg_duration' in df_dwell.columns:
            duration_stats = df_dwell['avg_duration']
            results['overall_stats'] = {
                'avg_duration': duration_stats.mean(),
                'max_duration': duration_stats.max(),
                'min_duration': duration_stats.min(),
                'std_duration': duration_stats.std()
            }
        
        # Calculate zone-specific metrics
        if 'zone_id' in df_dwell.columns:
            for zone_id, group in df_dwell.groupby('zone_id'):
                results['zone_metrics'][zone_id] = {
                    'avg_duration': group['avg_duration'].mean(),
                    'total_dwells': group['total_dwells'].sum(),
                    'unique_visitors': group['unique_visitors'].sum(),
                    'sample_count': len(group)
                }
        
        # Correlate with path data if available
        if len(self.path_data) >= self.min_data_points:
            df_path = pd.DataFrame(self.path_data)
            
            if 'zones_visited' in df_path.columns:
                zone_visit_counts = df_path['zones_visited'].value_counts().to_dict()
                results['zone_visit_distribution'] = zone_visit_counts
            
            # Calculate path-dwell correlation
            if 'path_duration' in df_path.columns and 'avg_duration' in df_dwell.columns:
                correlation = self._calculate_pearson_correlation(
                    df_path['path_duration'].tolist(),
                    df_dwell['avg_duration'].tolist()
                )
                if correlation:
                    results['path_dwell_correlation'] = correlation
        
        self._cache_result(cache_key, results)
        return results
    
    def get_path_correlations(self):
        """Calculate correlations between path patterns and other metrics"""
        cache_key = 'path'
        cached_result = self._check_cache_and_data(cache_key, self.path_data)
        if cached_result is not None:
            return cached_result
        
        df_path = pd.DataFrame(self.path_data)
        results = {'status': 'success', 'path_metrics': {}, 'transition_metrics': {}}
        
        # Calculate basic path statistics
        metric_columns = ['path_distance', 'path_duration', 'avg_velocity', 'zones_visited']
        for column in metric_columns:
            if column in df_path.columns:
                series = df_path[column]
                results['path_metrics'][f'avg_{column}'] = series.mean()
                results['path_metrics'][f'max_{column}'] = series.max()
                if column != 'zones_visited':  # zones_visited doesn't need min
                    results['path_metrics'][f'min_{column}'] = series.min()
        
        # Analyze zone transitions
        if self.temporal_data:
            df_temporal = pd.DataFrame(self.temporal_data)
            
            if 'source_zone' in df_temporal.columns and 'target_zone' in df_temporal.columns:
                transitions = (df_temporal.groupby(['source_zone', 'target_zone'])
                             ['transition_count'].sum().reset_index()
                             .sort_values('transition_count', ascending=False))
                
                # Convert to dictionary for top transitions
                top_transitions = {}
                for _, row in transitions.head(10).iterrows():
                    key = f"{row['source_zone']}→{row['target_zone']}"
                    top_transitions[key] = int(row['transition_count'])
                
                results['transition_metrics'] = {
                    'top_transitions': top_transitions,
                    'total_transitions': int(transitions['transition_count'].sum())
                }
        
        self._cache_result(cache_key, results)
        return results
    
    def get_temporal_correlations(self):
        """Calculate correlations based on time of day or day of week"""
        cache_key = 'temporal'
        
        # Check data sufficiency for both datasets
        if (len(self.dwell_data) < self.min_data_points or 
            len(self.path_data) < self.min_data_points):
            result = {
                'status': 'insufficient_data',
                'min_required': self.min_data_points,
                'dwell_count': len(self.dwell_data),
                'path_count': len(self.path_data)
            }
            self._cache_result(cache_key, result)
            return result
        
        cached_result = self._check_cache_and_data(cache_key, self.dwell_data)
        if cached_result is not None and cached_result.get('status') != 'insufficient_data':
            return cached_result
        
        # Add datetime columns to both datasets
        def add_time_components(data_list):
            result = []
            for entry in data_list:
                entry_copy = entry.copy()
                dt = pd.to_datetime(entry_copy['timestamp'], unit='s')
                entry_copy['datetime'] = dt
                entry_copy['hour'] = dt.hour
                entry_copy['day_of_week'] = dt.dayofweek
                result.append(entry_copy)
            return pd.DataFrame(result)
        
        df_dwell = add_time_components(self.dwell_data)
        df_path = add_time_components(self.path_data)
        
        results = {'status': 'success', 'hourly_metrics': {}, 'daily_metrics': {}}
        
        # Calculate hourly and daily metrics
        time_aggregations = [
            ('hour', 'hourly_metrics'),
            ('day_of_week', 'daily_metrics')
        ]
        
        for time_col, result_key in time_aggregations:
            if time_col in df_dwell.columns and 'avg_duration' in df_dwell.columns:
                hourly_dwell = df_dwell.groupby(time_col)['avg_duration'].mean().to_dict()
                results[result_key]['dwell_time'] = hourly_dwell
            
            if time_col in df_path.columns:
                for metric in ['path_duration', 'zones_visited']:
                    if metric in df_path.columns:
                        grouped = df_path.groupby(time_col)[metric].mean().to_dict()
                        results[result_key][metric] = grouped
        
        self._cache_result(cache_key, results)
        return results
    
    def get_sales_correlations(self):
        """Calculate correlations between sales data and customer behavior"""
        if not self.has_sales_data or len(self.sales_data) < self.min_data_points:
            return {
                'status': 'insufficient_data',
                'has_sales_data': self.has_sales_data,
                'sales_data_count': len(self.sales_data)
            }
        
        cache_key = 'sales'
        cached_result = self._check_cache_and_data(cache_key, self.sales_data)
        if cached_result is not None:
            return cached_result
        
        df_sales = pd.DataFrame(self.sales_data)
        df_dwell = pd.DataFrame(self.dwell_data)
        
        results = {'status': 'success', 'dwell_sales_correlation': None}
        
        # Calculate sales-dwell correlation if both datasets have required columns
        if ('amount' in df_sales.columns and 'avg_duration' in df_dwell.columns and
            len(df_sales) > 5 and len(df_dwell) > 5):
            
            # Convert to datetime and group by hour
            df_sales['datetime'] = pd.to_datetime(df_sales['timestamp'], unit='s')
            df_dwell['datetime'] = pd.to_datetime(df_dwell['timestamp'], unit='s')
            
            df_sales['hour'] = df_sales['datetime'].dt.floor('H')
            df_dwell['hour'] = df_dwell['datetime'].dt.floor('H')
            
            # Aggregate by hour and merge
            hourly_sales = df_sales.groupby('hour')['amount'].sum().reset_index()
            hourly_dwell = df_dwell.groupby('hour')['avg_duration'].mean().reset_index()
            merged = pd.merge(hourly_sales, hourly_dwell, on='hour', how='inner')
            
            if len(merged) > 5:
                correlation = self._calculate_pearson_correlation(
                    merged['amount'].tolist(),
                    merged['avg_duration'].tolist()
                )
                if correlation:
                    results['dwell_sales_correlation'] = correlation
        
        self._cache_result(cache_key, results)
        return results
    
    def get_all_correlations(self):
        """Get all correlation data in a single comprehensive report"""
        return {
            'demographic': self.get_demographic_correlations(),
            'dwell': self.get_dwell_correlations(),
            'path': self.get_path_correlations(),
            'temporal': self.get_temporal_correlations(),
            'sales': self.get_sales_correlations() if self.has_sales_data else {'status': 'no_sales_data'},
            'timestamp': time.time()
        }
    
    def clear_cache(self):
        """Clear the correlation cache"""
        self.correlation_cache = {}
        self.last_correlation_time = 0
        logger.debug("Correlation cache cleared")
    
    def generate_correlation_visualization(self, visualization_type='demographic'):
        """
        Generate visualization for correlation data
        
        Args:
            visualization_type (str): Type of visualization to generate
            
        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        plt.figure(figsize=(10, 6))
        
        correlations = getattr(self, f'get_{visualization_type}_correlations')()
        
        if correlations['status'] != 'success':
            plt.text(0.5, 0.5, f"Insufficient data: {correlations['status']}", 
                     horizontalalignment='center', verticalalignment='center')
            return plt.gcf()
        
        # Generate visualizations based on type
        if visualization_type == 'demographic':
            self._plot_demographic_visualization(correlations)
        elif visualization_type == 'dwell':
            self._plot_dwell_visualization(correlations)
        elif visualization_type == 'path':
            self._plot_path_visualization(correlations)
        elif visualization_type == 'temporal':
            self._plot_temporal_visualization(correlations)
        elif visualization_type == 'sales':
            self._plot_sales_visualization(correlations)
        
        plt.tight_layout()
        return plt.gcf()
    
    def _plot_demographic_visualization(self, correlations):
        """Plot demographic distribution visualizations"""
        distributions = ['gender_distribution', 'emotion_distribution', 
                        'race_distribution', 'age_distribution']
        
        for i, dist_key in enumerate(distributions, 1):
            if dist_key in correlations:
                plt.subplot(2, 2, i)
                data = correlations[dist_key]
                
                if dist_key == 'age_distribution':
                    plt.bar(data.keys(), data.values())
                    plt.title('Age Distribution')
                    plt.xticks(rotation=45)
                else:
                    plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%')
                    plt.title(dist_key.replace('_', ' ').title())
    
    def _plot_dwell_visualization(self, correlations):
        """Plot dwell time visualizations"""
        if 'zone_metrics' in correlations:
            zone_ids = list(correlations['zone_metrics'].keys())
            avg_durations = [stats['avg_duration'] for stats in correlations['zone_metrics'].values()]
            total_dwells = [stats['total_dwells'] for stats in correlations['zone_metrics'].values()]
            
            plt.subplot(121)
            plt.bar(zone_ids, avg_durations)
            plt.title('Average Dwell Duration by Zone')
            plt.xticks(rotation=45)
            
            plt.subplot(122)
            plt.bar(zone_ids, total_dwells)
            plt.title('Total Dwells by Zone')
            plt.xticks(rotation=45)
    
    def _plot_path_visualization(self, correlations):
        """Plot path analysis visualizations"""
        if ('transition_metrics' in correlations and 
            'top_transitions' in correlations['transition_metrics']):
            transitions = correlations['transition_metrics']['top_transitions']
            plt.bar(transitions.keys(), transitions.values())
            plt.title('Top Zone Transitions')
            plt.xticks(rotation=90)
    
    def _plot_temporal_visualization(self, correlations):
        """Plot temporal correlation visualizations"""
        if ('hourly_metrics' in correlations and 
            'dwell_time' in correlations['hourly_metrics']):
            plt.subplot(221)
            dwell_data = correlations['hourly_metrics']['dwell_time']
            hours = sorted(dwell_data.keys())
            values = [dwell_data[h] for h in hours]
            plt.plot(hours, values, 'o-')
            plt.title('Dwell Time by Hour')
            plt.xlabel('Hour of Day')
            plt.ylabel('Average Dwell Time (s)')
        
        if ('daily_metrics' in correlations and 
            'dwell_time' in correlations['daily_metrics']):
            plt.subplot(222)
            daily_data = correlations['daily_metrics']['dwell_time']
            days = sorted(daily_data.keys())
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            values = [daily_data[d] for d in days]
            plt.bar(days, values)
            plt.xticks(days, [day_names[d] for d in days])
            plt.title('Dwell Time by Day of Week')
            plt.ylabel('Average Dwell Time (s)')
    
    def _plot_sales_visualization(self, correlations):
        """Plot sales correlation visualizations"""
        if 'dwell_sales_correlation' in correlations and correlations['dwell_sales_correlation']:
            corr = correlations['dwell_sales_correlation']
            plt.text(0.5, 0.5, f"Dwell to Sales Correlation: {corr['pearson_r']:.2f}\n" +
                     f"p-value: {corr['p_value']:.4f}\n" +
                     f"Significant: {corr['significant']}", 
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=14)
            plt.title('Dwell Time vs Sales Correlation') 