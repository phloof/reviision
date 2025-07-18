"""
Correlation Analysis Module for Retail Analytics System

This module provides comprehensive statistical analysis and correlation detection
for customer behavior, demographics, path patterns, and temporal trends.
"""

import logging
import numpy as np
import hashlib
import json
import time
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional, Union

try:
    from scipy import stats
    from scipy.stats import pearsonr, spearmanr, chi2_contingency, ttest_ind, mannwhitneyu
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class CorrelationAnalyzer:
    """
    Advanced correlation analyzer for retail customer behavior analysis
    
    This class provides configurable statistical analysis including demographic
    correlations, temporal patterns, spatial analysis, and behavioral insights.
    """
    
    def __init__(self, config, database=None):
        """
        Initialize the correlation analyzer
        
        Args:
            config (dict): Configuration dictionary
            database: Database connection for data access
        """
        self.config = config
        self.database = database
        
        # Analysis configuration
        self.correlation_methods = config.get('methods', ['pearson', 'spearman', 'chi_square'])
        self.significance_threshold = config.get('significance_threshold', 0.05)
        self.min_sample_size = config.get('min_sample_size', 10)
        self.confidence_level = config.get('confidence_level', 0.95)
        
        # Temporal analysis settings
        self.temporal_windows = config.get('temporal_windows', [1, 6, 24, 168])  # hours
        self.day_comparison_enabled = config.get('day_comparison', True)
        self.seasonal_analysis = config.get('seasonal_analysis', True)
        self.trend_detection = config.get('trend_detection', True)
        
        # Demographic analysis settings
        self.demographic_correlations = config.get('demographic_correlations', True)
        self.zone_analysis = config.get('zone_analysis', True)
        self.path_analysis = config.get('path_analysis', True)
        self.behavioral_segmentation = config.get('behavioral_segmentation', True)
        
        # Caching settings
        self.cache_results = config.get('cache_results', True)
        self.cache_expiry = config.get('cache_expiry', 3600)  # seconds
        
        # Analysis cache
        self.analysis_cache = {}
        
        # Statistical thresholds
        self.effect_size_thresholds = {
            'small': 0.1,
            'medium': 0.3,
            'large': 0.5
        }
        
        logger.info(f"CorrelationAnalyzer initialized with methods: {self.correlation_methods}, "
                   f"min_sample_size: {self.min_sample_size}, "
                   f"significance_threshold: {self.significance_threshold}")
    
    def analyze_demographic_paths(self, time_range_hours=24):
        """
        Analyze correlations between demographics and path patterns
        
        Args:
            time_range_hours (int): Hours to analyze
            
        Returns:
            dict: Demographic-path correlation analysis
        """
        if not self.demographic_correlations or not self.database:
            return {'error': 'Demographic correlation analysis not available'}
        
        # Check cache first
        cache_key = f"demo_paths_{time_range_hours}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Get data from database
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_range_hours)
            
            # Query demographic and path data
            demographic_data = self._get_demographic_data(start_time, end_time)
            path_data = self._get_path_data(start_time, end_time)
            
            if len(demographic_data) < self.min_sample_size:
                return {'error': f'Insufficient data: {len(demographic_data)} samples (min: {self.min_sample_size})'}
            
            # Analyze correlations
            results = {
                'analysis_type': 'demographic_paths',
                'time_range': f'{time_range_hours} hours',
                'sample_size': len(demographic_data),
                'timestamp': datetime.now().isoformat(),
                'correlations': {}
            }
            
            # Gender vs Path Patterns
            if 'gender' in [d.get('gender') for d in demographic_data]:
                gender_path_corr = self._analyze_gender_path_correlation(demographic_data, path_data)
                results['correlations']['gender_paths'] = gender_path_corr
            
            # Age Group vs Path Patterns
            if 'age_group' in [d.get('age_group') for d in demographic_data]:
                age_path_corr = self._analyze_age_path_correlation(demographic_data, path_data)
                results['correlations']['age_paths'] = age_path_corr
            
            # Demographic vs Zone Preferences
            if self.zone_analysis:
                zone_demo_corr = self._analyze_demographic_zone_correlation(demographic_data, path_data)
                results['correlations']['demographic_zones'] = zone_demo_corr
            
            # Behavioral Segmentation
            if self.behavioral_segmentation:
                behavioral_segments = self._perform_behavioral_segmentation(demographic_data, path_data)
                results['behavioral_segments'] = behavioral_segments
            
            # Cache results
            self._cache_result(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in demographic-path analysis: {e}")
            return {'error': str(e)}
    
    def analyze_temporal_patterns(self, comparison_days=7, analysis_type='traffic'):
        """
        Analyze temporal patterns and day-to-day comparisons
        
        Args:
            comparison_days (int): Number of days to compare
            analysis_type (str): Type of analysis ('traffic', 'demographics', 'paths')
            
        Returns:
            dict: Temporal pattern analysis
        """
        if not self.day_comparison_enabled or not self.database:
            return {'error': 'Temporal analysis not available'}
        
        cache_key = f"temporal_{analysis_type}_{comparison_days}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            results = {
                'analysis_type': f'temporal_{analysis_type}',
                'comparison_days': comparison_days,
                'timestamp': datetime.now().isoformat(),
                'patterns': {},
                'correlations': {},
                'trends': {}
            }
            
            # Daily traffic patterns
            if analysis_type in ['traffic', 'all']:
                daily_traffic = self._analyze_daily_traffic_patterns(comparison_days)
                results['patterns']['daily_traffic'] = daily_traffic
                
                # Day-to-day correlations
                traffic_correlations = self._analyze_traffic_correlations(daily_traffic)
                results['correlations']['daily_traffic'] = traffic_correlations
            
            # Hourly patterns
            if analysis_type in ['hourly', 'all']:
                hourly_patterns = self._analyze_hourly_patterns(comparison_days)
                results['patterns']['hourly'] = hourly_patterns
            
            # Weekly trends
            if self.trend_detection and comparison_days >= 7:
                weekly_trends = self._analyze_weekly_trends(comparison_days)
                results['trends']['weekly'] = weekly_trends
            
            # Demographic temporal patterns
            if analysis_type in ['demographics', 'all'] and self.demographic_correlations:
                demo_temporal = self._analyze_demographic_temporal_patterns(comparison_days)
                results['patterns']['demographic_temporal'] = demo_temporal
            
            # Path temporal patterns
            if analysis_type in ['paths', 'all'] and self.path_analysis:
                path_temporal = self._analyze_path_temporal_patterns(comparison_days)
                results['patterns']['path_temporal'] = path_temporal
            
            # Statistical significance testing
            if 'daily_traffic' in results['patterns']:
                significance_tests = self._perform_temporal_significance_tests(
                    results['patterns']['daily_traffic']
                )
                results['statistical_tests'] = significance_tests
            
            # Cache results
            self._cache_result(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            return {'error': str(e)}
    
    def analyze_zone_preferences(self, demographic_filter=None, time_range_hours=24):
        """
        Analyze zone preferences by demographics
        
        Args:
            demographic_filter (dict): Filter criteria (e.g., {'gender': 'male'})
            time_range_hours (int): Hours to analyze
            
        Returns:
            dict: Zone preference analysis
        """
        if not self.zone_analysis or not self.database:
            return {'error': 'Zone analysis not available'}
        
        cache_key = f"zones_{json.dumps(demographic_filter, sort_keys=True)}_{time_range_hours}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            # Get zone visit data
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_range_hours)
            
            zone_data = self._get_zone_visit_data(start_time, end_time, demographic_filter)
            
            if len(zone_data) < self.min_sample_size:
                return {'error': f'Insufficient zone data: {len(zone_data)} visits'}
            
            results = {
                'analysis_type': 'zone_preferences',
                'demographic_filter': demographic_filter,
                'time_range': f'{time_range_hours} hours',
                'sample_size': len(zone_data),
                'timestamp': datetime.now().isoformat(),
                'zone_statistics': {},
                'correlations': {},
                'preferences': {}
            }
            
            # Zone visit statistics
            zone_stats = self._calculate_zone_statistics(zone_data)
            results['zone_statistics'] = zone_stats
            
            # Zone preference analysis by demographics
            if not demographic_filter:  # Analyze all demographics
                demo_zone_prefs = self._analyze_all_demographic_zone_preferences(zone_data)
                results['preferences'] = demo_zone_prefs
            
            # Zone transition patterns
            transition_patterns = self._analyze_zone_transitions(zone_data)
            results['transition_patterns'] = transition_patterns
            
            # Popular paths between zones
            inter_zone_paths = self._analyze_inter_zone_paths(zone_data)
            results['inter_zone_paths'] = inter_zone_paths
            
            # Cache results
            self._cache_result(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in zone preference analysis: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self, time_range_hours=24):
        """
        Generate comprehensive correlation analysis report
        
        Args:
            time_range_hours (int): Hours to analyze
            
        Returns:
            dict: Comprehensive analysis report
        """
        cache_key = f"comprehensive_{time_range_hours}"
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            report = {
                'report_type': 'comprehensive_correlation_analysis',
                'time_range': f'{time_range_hours} hours',
                'generated_at': datetime.now().isoformat(),
                'analyses': {},
                'summary': {},
                'insights': [],
                'recommendations': []
            }
            
            # Demographic-path analysis
            if self.demographic_correlations:
                demo_analysis = self.analyze_demographic_paths(time_range_hours)
                report['analyses']['demographic_paths'] = demo_analysis
            
            # Temporal patterns
            if self.day_comparison_enabled:
                temporal_analysis = self.analyze_temporal_patterns(
                    comparison_days=min(7, time_range_hours // 24)
                )
                report['analyses']['temporal_patterns'] = temporal_analysis
            
            # Zone preferences
            if self.zone_analysis:
                zone_analysis = self.analyze_zone_preferences(
                    time_range_hours=time_range_hours
                )
                report['analyses']['zone_preferences'] = zone_analysis
            
            # Generate insights and recommendations
            insights = self._generate_insights(report['analyses'])
            report['insights'] = insights
            
            recommendations = self._generate_recommendations(report['analyses'])
            report['recommendations'] = recommendations
            
            # Generate summary statistics
            summary = self._generate_summary_statistics(report['analyses'])
            report['summary'] = summary
            
            # Cache results
            self._cache_result(cache_key, report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {'error': str(e)}
    
    # Private helper methods
    
    def _get_demographic_data(self, start_time, end_time):
        """Get demographic data from database"""
        try:
            conn = self.database._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT d.person_id, d.age, ag.group_name as age_group, 
                       g.gender_name as gender, r.race_name as race,
                       e.emotion_name as emotion, d.confidence, d.timestamp
                FROM demographics d
                LEFT JOIN age_groups ag ON d.age_group_id = ag.id
                LEFT JOIN genders g ON d.gender_id = g.id
                LEFT JOIN races r ON d.race_id = r.id
                LEFT JOIN emotions e ON d.emotion_id = e.id
                WHERE d.timestamp >= ? AND d.timestamp <= ?
                AND d.confidence >= ?
            ''', (start_time, end_time, 0.5))
            
            demographics = []
            for row in cursor.fetchall():
                demographics.append({
                    'person_id': row[0],
                    'age': row[1],
                    'age_group': row[2],
                    'gender': row[3],
                    'race': row[4],
                    'emotion': row[5],
                    'confidence': row[6],
                    'timestamp': row[7]
                })
            
            return demographics
            
        except Exception as e:
            logger.error(f"Error getting demographic data: {e}")
            return []
    
    def _get_path_data(self, start_time, end_time):
        """Get path data from database"""
        try:
            conn = self.database._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT person_id, session_id, x_position, y_position,
                       timestamp, speed, movement_type, zone_name,
                       path_complexity
                FROM customer_paths
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY person_id, session_id, sequence_number
            ''', (start_time, end_time))
            
            paths = []
            for row in cursor.fetchall():
                paths.append({
                    'person_id': row[0],
                    'session_id': row[1],
                    'x': row[2],
                    'y': row[3],
                    'timestamp': row[4],
                    'speed': row[5],
                    'movement_type': row[6],
                    'zone_name': row[7],
                    'complexity': row[8]
                })
            
            return paths
            
        except Exception as e:
            logger.error(f"Error getting path data: {e}")
            return []
    
    def _analyze_gender_path_correlation(self, demographic_data, path_data):
        """Analyze correlation between gender and path patterns"""
        try:
            # Create gender-path mapping
            person_gender = {d['person_id']: d['gender'] for d in demographic_data 
                           if d['gender'] and d['gender'] != 'unknown'}
            
            # Group path data by gender
            gender_paths = defaultdict(list)
            for path in path_data:
                person_id = path['person_id']
                if person_id in person_gender:
                    gender = person_gender[person_id]
                    gender_paths[gender].append(path)
            
            results = {
                'method': 'categorical_analysis',
                'sample_sizes': {gender: len(paths) for gender, paths in gender_paths.items()},
                'path_metrics': {},
                'statistical_tests': {}
            }
            
            # Calculate path metrics by gender
            for gender, paths in gender_paths.items():
                if len(paths) >= self.min_sample_size:
                    speeds = [p['speed'] for p in paths if p['speed'] is not None]
                    complexities = [p['complexity'] for p in paths if p['complexity'] is not None]
                    
                    results['path_metrics'][gender] = {
                        'avg_speed': np.mean(speeds) if speeds else 0,
                        'avg_complexity': np.mean(complexities) if complexities else 0,
                        'path_count': len(paths),
                        'movement_types': Counter([p['movement_type'] for p in paths])
                    }
            
            # Statistical tests if we have sufficient data
            if len(gender_paths) >= 2 and SCIPY_AVAILABLE:
                genders = list(gender_paths.keys())
                if len(genders) == 2:
                    # T-test for speed differences
                    speeds1 = [p['speed'] for p in gender_paths[genders[0]] if p['speed'] is not None]
                    speeds2 = [p['speed'] for p in gender_paths[genders[1]] if p['speed'] is not None]
                    
                    if len(speeds1) >= 3 and len(speeds2) >= 3:
                        t_stat, p_value = ttest_ind(speeds1, speeds2)
                        results['statistical_tests']['speed_ttest'] = {
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < self.significance_threshold
                        }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in gender-path correlation: {e}")
            return {'error': str(e)}
    
    def _analyze_age_path_correlation(self, demographic_data, path_data):
        """Analyze correlation between age groups and path patterns"""
        try:
            # Create age-path mapping
            person_age_group = {d['person_id']: d['age_group'] for d in demographic_data 
                              if d['age_group'] and d['age_group'] != 'unknown'}
            
            # Group path data by age group
            age_paths = defaultdict(list)
            for path in path_data:
                person_id = path['person_id']
                if person_id in person_age_group:
                    age_group = person_age_group[person_id]
                    age_paths[age_group].append(path)
            
            results = {
                'method': 'age_group_analysis',
                'sample_sizes': {age: len(paths) for age, paths in age_paths.items()},
                'path_metrics': {},
                'correlations': {}
            }
            
            # Calculate metrics by age group
            for age_group, paths in age_paths.items():
                if len(paths) >= self.min_sample_size:
                    speeds = [p['speed'] for p in paths if p['speed'] is not None]
                    complexities = [p['complexity'] for p in paths if p['complexity'] is not None]
                    
                    results['path_metrics'][age_group] = {
                        'avg_speed': np.mean(speeds) if speeds else 0,
                        'avg_complexity': np.mean(complexities) if complexities else 0,
                        'path_count': len(paths)
                    }
            
            # ANOVA test if sufficient groups and data
            if len(age_paths) >= 3 and SCIPY_AVAILABLE:
                speed_groups = []
                for paths in age_paths.values():
                    speeds = [p['speed'] for p in paths if p['speed'] is not None]
                    if len(speeds) >= 3:
                        speed_groups.append(speeds)
                
                if len(speed_groups) >= 3:
                    f_stat, p_value = stats.f_oneway(*speed_groups)
                    results['correlations']['speed_anova'] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_value),
                        'significant': p_value < self.significance_threshold
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in age-path correlation: {e}")
            return {'error': str(e)}
    
    def _analyze_demographic_zone_correlation(self, demographic_data, path_data):
        """Analyze correlation between demographics and zone preferences"""
        try:
            # Create demographic-zone mapping
            person_demo = {d['person_id']: d for d in demographic_data}
            
            # Collect zone visits by demographics
            zone_visits = defaultdict(lambda: defaultdict(int))
            demo_zone_counts = defaultdict(lambda: defaultdict(int))
            
            for path in path_data:
                person_id = path['person_id']
                zone = path['zone_name']
                
                if person_id in person_demo and zone:
                    demo = person_demo[person_id]
                    gender = demo.get('gender', 'unknown')
                    age_group = demo.get('age_group', 'unknown')
                    
                    if gender != 'unknown':
                        demo_zone_counts[f"gender_{gender}"][zone] += 1
                    if age_group != 'unknown':
                        demo_zone_counts[f"age_{age_group}"][zone] += 1
            
            results = {
                'method': 'demographic_zone_correlation',
                'zone_preferences': {},
                'statistical_tests': {}
            }
            
            # Calculate zone preferences by demographic
            for demo_key, zone_counts in demo_zone_counts.items():
                total_visits = sum(zone_counts.values())
                if total_visits >= self.min_sample_size:
                    preferences = {zone: count/total_visits 
                                 for zone, count in zone_counts.items()}
                    results['zone_preferences'][demo_key] = {
                        'preferences': preferences,
                        'total_visits': total_visits,
                        'most_preferred': max(preferences, key=preferences.get),
                        'least_preferred': min(preferences, key=preferences.get)
                    }
            
            # Chi-square test for independence
            if len(demo_zone_counts) >= 2 and SCIPY_AVAILABLE:
                # Create contingency table for chi-square test
                all_zones = set()
                for zone_counts in demo_zone_counts.values():
                    all_zones.update(zone_counts.keys())
                
                if len(all_zones) >= 2:
                    contingency_table = []
                    demo_labels = []
                    
                    for demo_key, zone_counts in demo_zone_counts.items():
                        row = [zone_counts.get(zone, 0) for zone in sorted(all_zones)]
                        if sum(row) >= 5:  # Minimum expected frequency
                            contingency_table.append(row)
                            demo_labels.append(demo_key)
                    
                    if len(contingency_table) >= 2:
                        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                        results['statistical_tests']['chi_square'] = {
                            'chi2_statistic': float(chi2_stat),
                            'p_value': float(p_value),
                            'degrees_of_freedom': int(dof),
                            'significant': p_value < self.significance_threshold
                        }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in demographic-zone correlation: {e}")
            return {'error': str(e)}
    
    def _perform_behavioral_segmentation(self, demographic_data, path_data):
        """Perform behavioral segmentation using clustering"""
        if not SCIPY_AVAILABLE:
            return {'error': 'SciPy not available for clustering'}
        
        try:
            # Create feature vectors for each person
            person_features = {}
            person_demo = {d['person_id']: d for d in demographic_data}
            
            # Group paths by person
            person_paths = defaultdict(list)
            for path in path_data:
                person_paths[path['person_id']].append(path)
            
            # Extract behavioral features
            for person_id, paths in person_paths.items():
                if len(paths) >= 3 and person_id in person_demo:  # Minimum paths for analysis
                    speeds = [p['speed'] for p in paths if p['speed'] is not None]
                    complexities = [p['complexity'] for p in paths if p['complexity'] is not None]
                    zones = [p['zone_name'] for p in paths if p['zone_name']]
                    
                    features = [
                        np.mean(speeds) if speeds else 0,          # Avg speed
                        np.std(speeds) if len(speeds) > 1 else 0,  # Speed variance
                        np.mean(complexities) if complexities else 0,  # Avg complexity
                        len(set(zones)) if zones else 0,           # Zone diversity
                        len(paths)                                 # Total path points
                    ]
                    
                    person_features[person_id] = features
            
            if len(person_features) < self.min_sample_size:
                return {'error': f'Insufficient data for segmentation: {len(person_features)} people'}
            
            # Perform K-means clustering
            features_array = np.array(list(person_features.values()))
            person_ids = list(person_features.keys())
            
            # Determine optimal number of clusters (2-5)
            n_clusters = min(5, max(2, len(person_features) // 10))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_array)
            
            # Analyze segments
            segments = defaultdict(list)
            for i, person_id in enumerate(person_ids):
                cluster_id = cluster_labels[i]
                segments[cluster_id].append({
                    'person_id': person_id,
                    'features': person_features[person_id],
                    'demographics': person_demo.get(person_id, {})
                })
            
            # Characterize each segment
            segment_analysis = {}
            for cluster_id, people in segments.items():
                features_matrix = np.array([p['features'] for p in people])
                
                segment_analysis[f'segment_{cluster_id}'] = {
                    'size': len(people),
                    'characteristics': {
                        'avg_speed': float(np.mean(features_matrix[:, 0])),
                        'speed_variance': float(np.mean(features_matrix[:, 1])),
                        'avg_complexity': float(np.mean(features_matrix[:, 2])),
                        'zone_diversity': float(np.mean(features_matrix[:, 3])),
                        'path_activity': float(np.mean(features_matrix[:, 4]))
                    },
                    'demographics': self._analyze_segment_demographics(people)
                }
            
            return {
                'method': 'kmeans_clustering',
                'n_clusters': n_clusters,
                'total_people': len(person_features),
                'segments': segment_analysis,
                'cluster_centers': kmeans.cluster_centers_.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in behavioral segmentation: {e}")
            return {'error': str(e)}
    
    def _analyze_segment_demographics(self, people):
        """Analyze demographic composition of a behavioral segment"""
        genders = [p['demographics'].get('gender') for p in people]
        age_groups = [p['demographics'].get('age_group') for p in people]
        
        return {
            'gender_distribution': dict(Counter([g for g in genders if g and g != 'unknown'])),
            'age_distribution': dict(Counter([a for a in age_groups if a and a != 'unknown'])),
            'dominant_gender': Counter(genders).most_common(1)[0][0] if genders else 'unknown',
            'dominant_age_group': Counter(age_groups).most_common(1)[0][0] if age_groups else 'unknown'
        }
    
    def _analyze_daily_traffic_patterns(self, comparison_days):
        """Analyze daily traffic patterns"""
        try:
            conn = self.database._get_connection()
            cursor = conn.cursor()
            
            # Get daily visitor counts
            cursor.execute('''
                SELECT DATE(timestamp) as day, COUNT(DISTINCT person_id) as visitors
                FROM detections
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY DATE(timestamp)
                ORDER BY day
            '''.format(comparison_days))
            
            daily_data = []
            for row in cursor.fetchall():
                daily_data.append({
                    'date': row[0],
                    'visitors': row[1],
                    'day_of_week': datetime.fromisoformat(row[0]).strftime('%A')
                })
            
            # Calculate statistics
            visitors = [d['visitors'] for d in daily_data]
            
            return {
                'daily_data': daily_data,
                'statistics': {
                    'avg_daily_visitors': float(np.mean(visitors)) if visitors else 0,
                    'max_daily_visitors': max(visitors) if visitors else 0,
                    'min_daily_visitors': min(visitors) if visitors else 0,
                    'std_daily_visitors': float(np.std(visitors)) if len(visitors) > 1 else 0
                },
                'day_of_week_analysis': self._analyze_day_of_week_patterns(daily_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing daily traffic: {e}")
            return {'error': str(e)}
    
    def _analyze_day_of_week_patterns(self, daily_data):
        """Analyze patterns by day of week"""
        dow_visitors = defaultdict(list)
        
        for day_data in daily_data:
            dow = day_data['day_of_week']
            dow_visitors[dow].append(day_data['visitors'])
        
        dow_stats = {}
        for dow, visitors in dow_visitors.items():
            dow_stats[dow] = {
                'avg_visitors': float(np.mean(visitors)),
                'count': len(visitors)
            }
        
        return dow_stats
    
    def _analyze_hourly_patterns(self, comparison_days):
        """Analyze hourly traffic patterns"""
        try:
            conn = self.database._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT strftime('%H', timestamp) as hour, COUNT(DISTINCT person_id) as visitors
                FROM detections
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            '''.format(comparison_days))
            
            hourly_data = []
            for row in cursor.fetchall():
                hourly_data.append({
                    'hour': int(row[0]),
                    'visitors': row[1]
                })
            
            # Find peak hours
            visitors = [d['visitors'] for d in hourly_data]
            if visitors:
                max_visitors = max(visitors)
                peak_hours = [d['hour'] for d in hourly_data if d['visitors'] == max_visitors]
            else:
                peak_hours = []
            
            return {
                'hourly_data': hourly_data,
                'peak_hours': peak_hours,
                'total_hours_analyzed': len(hourly_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing hourly patterns: {e}")
            return {'error': str(e)}
    
    def _cache_result(self, cache_key, result):
        """Cache analysis result"""
        if self.cache_results:
            cache_entry = {
                'result': result,
                'timestamp': time.time(),
                'expires_at': time.time() + self.cache_expiry
            }
            self.analysis_cache[cache_key] = cache_entry
            
            # Store in database cache if available
            if self.database:
                try:
                    parameters_hash = hashlib.md5(cache_key.encode()).hexdigest()
                    self.database.store_correlation_result(
                        analysis_type='correlation_analysis',
                        analysis_key=cache_key,
                        parameters_hash=parameters_hash,
                        result_data=json.dumps(result),
                        cache_hours=self.cache_expiry // 3600
                    )
                except Exception as e:
                    logger.debug(f"Error storing cache to database: {e}")
    
    def _get_cached_result(self, cache_key):
        """Get cached analysis result"""
        if not self.cache_results:
            return None
        
        # Check memory cache first
        if cache_key in self.analysis_cache:
            cache_entry = self.analysis_cache[cache_key]
            if time.time() < cache_entry['expires_at']:
                return cache_entry['result']
            else:
                del self.analysis_cache[cache_key]
        
        # Check database cache
        if self.database:
            try:
                parameters_hash = hashlib.md5(cache_key.encode()).hexdigest()
                cached_result = self.database.get_correlation_result(
                    analysis_type='correlation_analysis',
                    analysis_key=cache_key,
                    parameters_hash=parameters_hash
                )
                if cached_result:
                    return json.loads(cached_result['result_data'])
            except Exception as e:
                logger.debug(f"Error retrieving cache from database: {e}")
        
        return None
    
    def _generate_insights(self, analyses):
        """Generate insights from analysis results"""
        insights = []
        
        # Demographic insights
        if 'demographic_paths' in analyses:
            demo_analysis = analyses['demographic_paths']
            if 'correlations' in demo_analysis and 'gender_paths' in demo_analysis['correlations']:
                gender_data = demo_analysis['correlations']['gender_paths']
                if 'path_metrics' in gender_data:
                    insights.append({
                        'type': 'demographic',
                        'category': 'gender_behavior',
                        'insight': self._generate_gender_insight(gender_data['path_metrics'])
                    })
        
        # Temporal insights
        if 'temporal_patterns' in analyses:
            temporal_analysis = analyses['temporal_patterns']
            if 'patterns' in temporal_analysis and 'hourly' in temporal_analysis['patterns']:
                hourly_data = temporal_analysis['patterns']['hourly']
                insights.append({
                    'type': 'temporal',
                    'category': 'peak_hours',
                    'insight': f"Peak traffic hours: {', '.join(map(str, hourly_data.get('peak_hours', [])))}"
                })
        
        return insights
    
    def _generate_recommendations(self, analyses):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Zone optimization recommendations
        if 'zone_preferences' in analyses:
            zone_analysis = analyses['zone_preferences']
            if 'zone_statistics' in zone_analysis:
                recommendations.append({
                    'category': 'layout_optimization',
                    'priority': 'high',
                    'recommendation': 'Consider redistributing products based on zone visit patterns'
                })
        
        # Staffing recommendations based on temporal patterns
        if 'temporal_patterns' in analyses:
            recommendations.append({
                'category': 'staffing',
                'priority': 'medium',
                'recommendation': 'Adjust staff schedules based on peak hour analysis'
            })
        
        return recommendations
    
    def _generate_summary_statistics(self, analyses):
        """Generate summary statistics across all analyses"""
        summary = {
            'total_analyses': len(analyses),
            'analysis_types': list(analyses.keys()),
            'significant_correlations': 0,
            'sample_sizes': {}
        }
        
        # Count significant correlations
        for analysis_name, analysis_data in analyses.items():
            if isinstance(analysis_data, dict):
                if 'sample_size' in analysis_data:
                    summary['sample_sizes'][analysis_name] = analysis_data['sample_size']
                
                # Count statistical significance
                if 'statistical_tests' in analysis_data:
                    for test_name, test_result in analysis_data['statistical_tests'].items():
                        if isinstance(test_result, dict) and test_result.get('significant'):
                            summary['significant_correlations'] += 1
        
        return summary
    
    def _generate_gender_insight(self, gender_metrics):
        """Generate insight from gender path metrics"""
        if len(gender_metrics) < 2:
            return "Insufficient data for gender comparison"
        
        speeds = {gender: metrics['avg_speed'] for gender, metrics in gender_metrics.items()}
        fastest_gender = max(speeds, key=speeds.get)
        slowest_gender = min(speeds, key=speeds.get)
        
        return f"{fastest_gender.title()} customers move {speeds[fastest_gender]/speeds[slowest_gender]:.1f}x faster on average than {slowest_gender} customers"
    
    def cleanup_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.analysis_cache.items()
            if current_time >= entry['expires_at']
        ]
        
        for key in expired_keys:
            del self.analysis_cache[key]
        
        if self.database:
            try:
                self.database.cleanup_expired_correlations()
            except Exception as e:
                logger.debug(f"Error cleaning database cache: {e}")
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries") 