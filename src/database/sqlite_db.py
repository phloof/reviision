"""
SQLite Database implementation for Retail Analytics System
"""

import os
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class SQLiteDatabase:
    """
    SQLite database implementation for storing analytics data
    
    This class provides methods to store and retrieve analytics data
    including person detections, demographics, paths, and dwell times.
    """
    
    def __init__(self, config):
        """
        Initialize SQLite database connection
        
        Args:
            config (dict): Database configuration dictionary
        """
        self.config = config
        self.db_path = config.get('path', ':memory:')
        self.conn = None
        self.cursor = None
        
        # Initialize database
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables"""
        try:
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create tables
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    person_id INTEGER,
                    x1 INTEGER,
                    y1 INTEGER,
                    x2 INTEGER,
                    y2 INTEGER,
                    confidence REAL
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS demographics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    timestamp DATETIME,
                    age INTEGER,
                    gender TEXT,
                    race TEXT,
                    emotion TEXT,
                    confidence REAL
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS paths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    timestamp DATETIME,
                    x INTEGER,
                    y INTEGER,
                    zone_id TEXT
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS dwell_times (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    zone_id TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    duration REAL
                )
            ''')
            
            self.conn.commit()
            logger.info(f"SQLite database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
            raise
    
    def store_detection(self, person_id, bbox, confidence, timestamp=None):
        """
        Store a person detection
        
        Args:
            person_id (int): Person track ID
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            confidence (float): Detection confidence
            timestamp (datetime): Detection timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            self.cursor.execute('''
                INSERT INTO detections (timestamp, person_id, x1, y1, x2, y2, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, person_id, bbox[0], bbox[1], bbox[2], bbox[3], confidence))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing detection: {e}")
    
    def store_demographics(self, person_id, demographics, timestamp=None):
        """
        Store demographic information
        
        Args:
            person_id (int): Person track ID
            demographics (dict): Demographic information
            timestamp (datetime): Analysis timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            self.cursor.execute('''
                INSERT INTO demographics (person_id, timestamp, age, gender, race, emotion, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (person_id, timestamp, demographics.get('age'), demographics.get('gender'),
                 demographics.get('race'), demographics.get('emotion'), demographics.get('confidence')))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing demographics: {e}")
    
    def store_path_point(self, person_id, x, y, zone_id=None, timestamp=None):
        """
        Store a path point
        
        Args:
            person_id (int): Person track ID
            x (int): X coordinate
            y (int): Y coordinate
            zone_id (str): Zone identifier
            timestamp (datetime): Point timestamp
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            self.cursor.execute('''
                INSERT INTO paths (person_id, timestamp, x, y, zone_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (person_id, timestamp, x, y, zone_id))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing path point: {e}")
    
    def store_dwell_time(self, person_id, zone_id, start_time, end_time, duration):
        """
        Store dwell time information
        
        Args:
            person_id (int): Person track ID
            zone_id (str): Zone identifier
            start_time (datetime): Start time
            end_time (datetime): End time
            duration (float): Duration in seconds
        """
        try:
            self.cursor.execute('''
                INSERT INTO dwell_times (person_id, zone_id, start_time, end_time, duration)
                VALUES (?, ?, ?, ?, ?)
            ''', (person_id, zone_id, start_time, end_time, duration))
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error storing dwell time: {e}")
    
    def get_detections(self, start_time=None, end_time=None, person_id=None):
        """
        Get stored detections
        
        Args:
            start_time (datetime): Start time filter
            end_time (datetime): End time filter
            person_id (int): Person ID filter
            
        Returns:
            list: List of detection records
        """
        query = "SELECT * FROM detections WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        if person_id is not None:
            query += " AND person_id = ?"
            params.append(person_id)
            
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting detections: {e}")
            return []
    
    def get_demographics(self, start_time=None, end_time=None, person_id=None):
        """
        Get stored demographics
        
        Args:
            start_time (datetime): Start time filter
            end_time (datetime): End time filter
            person_id (int): Person ID filter
            
        Returns:
            list: List of demographic records
        """
        query = "SELECT * FROM demographics WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        if person_id is not None:
            query += " AND person_id = ?"
            params.append(person_id)
            
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting demographics: {e}")
            return []
    
    def get_paths(self, start_time=None, end_time=None, person_id=None):
        """
        Get stored paths
        
        Args:
            start_time (datetime): Start time filter
            end_time (datetime): End time filter
            person_id (int): Person ID filter
            
        Returns:
            list: List of path records
        """
        query = "SELECT * FROM paths WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        if person_id is not None:
            query += " AND person_id = ?"
            params.append(person_id)
            
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting paths: {e}")
            return []
    
    def get_dwell_times(self, start_time=None, end_time=None, person_id=None):
        """
        Get stored dwell times
        
        Args:
            start_time (datetime): Start time filter
            end_time (datetime): End time filter
            person_id (int): Person ID filter
            
        Returns:
            list: List of dwell time records
        """
        query = "SELECT * FROM dwell_times WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND start_time >= ?"
            params.append(start_time)
        if end_time:
            query += " AND end_time <= ?"
            params.append(end_time)
        if person_id is not None:
            query += " AND person_id = ?"
            params.append(person_id)
            
        try:
            self.cursor.execute(query, params)
            return self.cursor.fetchall()
        except Exception as e:
            logger.error(f"Error getting dwell times: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            logger.info("Database connection closed") 