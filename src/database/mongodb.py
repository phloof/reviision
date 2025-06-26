"""
MongoDB Database implementation for Retail Analytics System
"""

import os
import logging
from datetime import datetime
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

class MongoDBDatabase:
    """
    MongoDB database implementation for storing analytics data
    
    This class provides methods to store and retrieve analytics data
    including person detections, demographics, paths, and dwell times.
    """
    
    def __init__(self, config):
        """
        Initialize MongoDB database connection
        
        Args:
            config (dict): Database configuration dictionary
        """
        self.config = config
        self.uri = config.get('uri', 'mongodb://localhost:27017')
        self.db_name = config.get('database', 'reviision')
        self.client = None
        self.db = None
        
        # Initialize database
        self._init_db()
        
    def _init_db(self):
        """Initialize database connection and collections"""
        try:
            # Connect to MongoDB
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]
            
            # Create collections if they don't exist
            collections = ['detections', 'demographics', 'paths', 'dwell_times']
            for collection in collections:
                if collection not in self.db.list_collection_names():
                    self.db.create_collection(collection)
            
            # Create indexes
            self.db.detections.create_index([('timestamp', 1)])
            self.db.detections.create_index([('person_id', 1)])
            self.db.demographics.create_index([('timestamp', 1)])
            self.db.demographics.create_index([('person_id', 1)])
            self.db.paths.create_index([('timestamp', 1)])
            self.db.paths.create_index([('person_id', 1)])
            self.db.dwell_times.create_index([('start_time', 1)])
            self.db.dwell_times.create_index([('person_id', 1)])
            
            logger.info(f"MongoDB database initialized at {self.uri}/{self.db_name}")
            
        except Exception as e:
            logger.error(f"Error initializing MongoDB database: {e}")
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
            detection = {
                'timestamp': timestamp,
                'person_id': person_id,
                'x1': bbox[0],
                'y1': bbox[1],
                'x2': bbox[2],
                'y2': bbox[3],
                'confidence': confidence
            }
            self.db.detections.insert_one(detection)
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
            demo_data = {
                'person_id': person_id,
                'timestamp': timestamp,
                'age': demographics.get('age'),
                'gender': demographics.get('gender'),
                'race': demographics.get('race'),
                'emotion': demographics.get('emotion'),
                'confidence': demographics.get('confidence')
            }
            self.db.demographics.insert_one(demo_data)
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
            path_point = {
                'person_id': person_id,
                'timestamp': timestamp,
                'x': x,
                'y': y,
                'zone_id': zone_id
            }
            self.db.paths.insert_one(path_point)
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
            dwell_time = {
                'person_id': person_id,
                'zone_id': zone_id,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            }
            self.db.dwell_times.insert_one(dwell_time)
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
        query = {}
        
        if start_time or end_time:
            query['timestamp'] = {}
            if start_time:
                query['timestamp']['$gte'] = start_time
            if end_time:
                query['timestamp']['$lte'] = end_time
        if person_id is not None:
            query['person_id'] = person_id
            
        try:
            return list(self.db.detections.find(query))
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
        query = {}
        
        if start_time or end_time:
            query['timestamp'] = {}
            if start_time:
                query['timestamp']['$gte'] = start_time
            if end_time:
                query['timestamp']['$lte'] = end_time
        if person_id is not None:
            query['person_id'] = person_id
            
        try:
            return list(self.db.demographics.find(query))
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
        query = {}
        
        if start_time or end_time:
            query['timestamp'] = {}
            if start_time:
                query['timestamp']['$gte'] = start_time
            if end_time:
                query['timestamp']['$lte'] = end_time
        if person_id is not None:
            query['person_id'] = person_id
            
        try:
            return list(self.db.paths.find(query))
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
        query = {}
        
        if start_time or end_time:
            query['start_time'] = {}
            if start_time:
                query['start_time']['$gte'] = start_time
            if end_time:
                query['end_time'] = {'$lte': end_time}
        if person_id is not None:
            query['person_id'] = person_id
            
        try:
            return list(self.db.dwell_times.find(query))
        except Exception as e:
            logger.error(f"Error getting dwell times: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("Database connection closed") 