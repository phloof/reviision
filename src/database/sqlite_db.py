"""
SQLite Database implementation for Retail Analytics System
"""

import os
import sqlite3
import logging
import threading
from pathlib import Path
from datetime import datetime
import json

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
        self._local = threading.local()
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_db()
        
    def _get_connection(self):
        """Get a thread-safe database connection"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=20.0
            )
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute('PRAGMA synchronous=NORMAL')
            self._local.conn.execute('PRAGMA cache_size=10000')
            self._local.conn.execute('PRAGMA temp_store=memory')
        return self._local.conn
    
    def _init_db(self):
        """Initialize database tables with 3NF-compliant schema"""
        try:
            # Use the main connection for initialization
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=20.0
            )
            self.conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key constraints
            self.cursor = self.conn.cursor()
            
            # Create lookup tables first (for foreign key references)
            
            # Age groups lookup table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS age_groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_name VARCHAR(20) NOT NULL UNIQUE,
                    min_age INTEGER NOT NULL,
                    max_age INTEGER NOT NULL,
                    display_order INTEGER NOT NULL
                )
            ''')
            
            # Genders lookup table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS genders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gender_name VARCHAR(20) NOT NULL UNIQUE,
                    display_name VARCHAR(20) NOT NULL,
                    icon_class VARCHAR(50)
                )
            ''')
            
            # Emotions lookup table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    emotion_name VARCHAR(20) NOT NULL UNIQUE,
                    display_name VARCHAR(20) NOT NULL,
                    icon_class VARCHAR(50),
                    color_class VARCHAR(20)
                )
            ''')
            
            # Races lookup table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS races (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_name VARCHAR(30) NOT NULL UNIQUE,
                    display_name VARCHAR(30) NOT NULL
                )
            ''')
            
            # Persons table (normalized entity for individuals)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    first_detected DATETIME NOT NULL,
                    last_detected DATETIME NOT NULL,
                    total_visits INTEGER DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Detections table (normalized with foreign keys)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL,
                    y2 INTEGER NOT NULL,
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    camera_id VARCHAR(50),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                )
            ''')
            
            # Demographics table (3NF compliant)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS demographics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    detection_id INTEGER,
                    timestamp DATETIME NOT NULL,
                    age INTEGER CHECK (age >= 0 AND age <= 120),
                    age_group_id INTEGER,
                    gender_id INTEGER NOT NULL,
                    race_id INTEGER,
                    emotion_id INTEGER,
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    analysis_model VARCHAR(50),
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
                    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE SET NULL,
                    FOREIGN KEY (age_group_id) REFERENCES age_groups(id),
                    FOREIGN KEY (gender_id) REFERENCES genders(id),
                    FOREIGN KEY (race_id) REFERENCES races(id),
                    FOREIGN KEY (emotion_id) REFERENCES emotions(id)
                )
            ''')
            
            # Dwell times table (previously missing)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS dwell_times (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    detection_id INTEGER,
                    zone_name VARCHAR(100),
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    total_time REAL, -- in seconds
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
                    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE SET NULL
                )
            ''')
            
            # Analytics sessions table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_start DATETIME NOT NULL,
                    session_end DATETIME,
                    total_detections INTEGER DEFAULT 0,
                    total_persons INTEGER DEFAULT 0,
                    processing_time_ms INTEGER,
                    model_version VARCHAR(50),
                    status VARCHAR(20) DEFAULT 'active',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # User management tables (already 3NF compliant, keeping existing)
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    role TEXT DEFAULT 'viewer',
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_login DATETIME,
                    failed_login_attempts INTEGER DEFAULT 0,
                    locked_until DATETIME,
                    password_changed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_token TEXT,
                    session_expires_at DATETIME
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS login_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    success BOOLEAN,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    failure_reason TEXT
                )
            ''')
            
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_token TEXT UNIQUE NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create performance indexes
            indexes = [
                # Detection performance
                'CREATE INDEX IF NOT EXISTS idx_detections_person_timestamp ON detections(person_id, timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_detections_confidence ON detections(confidence)',
                
                # Demographics performance
                'CREATE INDEX IF NOT EXISTS idx_demographics_person_timestamp ON demographics(person_id, timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_demographics_timestamp ON demographics(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_demographics_age_group ON demographics(age_group_id)',
                'CREATE INDEX IF NOT EXISTS idx_demographics_gender ON demographics(gender_id)',
                
                # Dwell times performance
                'CREATE INDEX IF NOT EXISTS idx_dwell_times_person ON dwell_times(person_id)',
                'CREATE INDEX IF NOT EXISTS idx_dwell_times_start_time ON dwell_times(start_time)',
                'CREATE INDEX IF NOT EXISTS idx_dwell_times_active ON dwell_times(is_active)',
                
                # User management indexes
                'CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)',
                'CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)',
                'CREATE INDEX IF NOT EXISTS idx_users_session_token ON users(session_token)',
                'CREATE INDEX IF NOT EXISTS idx_login_attempts_username ON login_attempts(username)',
                'CREATE INDEX IF NOT EXISTS idx_login_attempts_ip ON login_attempts(ip_address)',
                'CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(session_token)',
                'CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)'
            ]
            
            for index_sql in indexes:
                self.cursor.execute(index_sql)
            
            # Populate lookup tables with default data if they're empty
            self._populate_lookup_tables()
            
            # Create default users for RBAC demonstration
            self._create_default_users()

            self.conn.commit()
            logger.info(f"3NF-compliant SQLite database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
            raise
    
    def _populate_lookup_tables(self):
        """Populate lookup tables with default data if they're empty"""
        try:
            # Populate age groups
            self.cursor.execute('SELECT COUNT(*) FROM age_groups')
            if self.cursor.fetchone()[0] == 0:
                age_groups_data = [
                    ('Under 18', 0, 17, 1),
                    ('18-24', 18, 24, 2),
                    ('25-34', 25, 34, 3),
                    ('35-44', 35, 44, 4),
                    ('45-54', 45, 54, 5),
                    ('55-64', 55, 64, 6),
                    ('65+', 65, 120, 7)
                ]
                self.cursor.executemany(
                    'INSERT INTO age_groups (group_name, min_age, max_age, display_order) VALUES (?, ?, ?, ?)',
                    age_groups_data
                )
            
            # Populate genders
            self.cursor.execute('SELECT COUNT(*) FROM genders')
            if self.cursor.fetchone()[0] == 0:
                genders_data = [
                    ('male', 'Male', 'fas fa-mars'),
                    ('female', 'Female', 'fas fa-venus'),
                    ('unknown', 'Unknown', 'fas fa-question')
                ]
                self.cursor.executemany(
                    'INSERT INTO genders (gender_name, display_name, icon_class) VALUES (?, ?, ?)',
                    genders_data
                )
            
            # Populate emotions
            self.cursor.execute('SELECT COUNT(*) FROM emotions')
            if self.cursor.fetchone()[0] == 0:
                emotions_data = [
                    ('happy', 'Happy', 'fas fa-smile', 'success'),
                    ('sad', 'Sad', 'fas fa-frown', 'info'),
                    ('angry', 'Angry', 'fas fa-angry', 'danger'),
                    ('surprised', 'Surprised', 'fas fa-surprise', 'warning'),
                    ('neutral', 'Neutral', 'fas fa-meh', 'secondary'),
                    ('focused', 'Focused', 'fas fa-eye', 'primary'),
                    ('tired', 'Tired', 'fas fa-tired', 'dark')
                ]
                self.cursor.executemany(
                    'INSERT INTO emotions (emotion_name, display_name, icon_class, color_class) VALUES (?, ?, ?, ?)',
                    emotions_data
                )
            
            # Populate races
            self.cursor.execute('SELECT COUNT(*) FROM races')
            if self.cursor.fetchone()[0] == 0:
                races_data = [
                    ('white', 'White'),
                    ('black', 'Black'),
                    ('asian', 'Asian'),
                    ('hispanic', 'Hispanic'),
                    ('other', 'Other'),
                    ('unknown', 'Unknown')
                ]
                self.cursor.executemany(
                    'INSERT INTO races (race_name, display_name) VALUES (?, ?)',
                    races_data
                )
            
        except Exception as e:
            logger.error(f"Error populating lookup tables: {e}")
            raise
    
    def update_user_password(self, user_id, new_password_hash):
        """
        Update user's password hash and timestamp
        
        Args:
            user_id (int): User ID to update
            new_password_hash (str): New hashed password
            
        Returns:
            bool: True if update successful, False otherwise
        """
        try:
            self.cursor.execute('''
                UPDATE users 
                SET password_hash = ?, password_changed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (new_password_hash, user_id))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating user password: {e}")
            return False
    
    def store_detection(self, person_id, bbox, confidence, timestamp=None, camera_id=None):
        """
        Store a person detection using 3NF schema
        
        Args:
            person_id (int): Person track ID
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            confidence (float): Detection confidence
            timestamp (datetime): Detection timestamp
            camera_id (str): Camera identifier
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Ensure person exists in persons table
            cursor.execute('SELECT id FROM persons WHERE id = ?', (person_id,))
            if not cursor.fetchone():
                # Create new person record
                cursor.execute('''
                    INSERT INTO persons (id, first_detected, last_detected, total_visits)
                    VALUES (?, ?, ?, 1)
                ''', (person_id, timestamp, timestamp))
            else:
                # Update last detected time and increment visits
                cursor.execute('''
                    UPDATE persons 
                    SET last_detected = ?, total_visits = total_visits + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (timestamp, person_id))
            
            # Insert detection record
            cursor.execute('''
                INSERT INTO detections (person_id, timestamp, x1, y1, x2, y2, confidence, camera_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (person_id, timestamp, bbox[0], bbox[1], bbox[2], bbox[3], confidence, camera_id))
            
            detection_id = cursor.lastrowid
            conn.commit()
            return detection_id
            
        except Exception as e:
            logger.error(f"Error storing detection: {e}")
            return None
    
    def store_demographics(self, person_id, demographics, timestamp=None, detection_id=None, analysis_model=None):
        """
        Store demographic information using 3NF schema with lookup tables
        
        Args:
            person_id (int): Person track ID
            demographics (dict): Demographic information
            timestamp (datetime): Analysis timestamp
            detection_id (int): Related detection ID
            analysis_model (str): Analysis model used
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get age and determine age group
            age = demographics.get('age')
            age_group_id = None
            if age is not None:
                cursor.execute('''
                    SELECT id FROM age_groups 
                    WHERE ? >= min_age AND ? <= max_age
                ''', (age, age))
                age_group_result = cursor.fetchone()
                if age_group_result:
                    age_group_id = age_group_result[0]
            
            # Get gender ID
            gender_name = demographics.get('gender', 'unknown').lower()
            cursor.execute('SELECT id FROM genders WHERE gender_name = ?', (gender_name,))
            gender_result = cursor.fetchone()
            gender_id = gender_result[0] if gender_result else None
            
            # If gender not found, default to unknown
            if gender_id is None:
                cursor.execute('SELECT id FROM genders WHERE gender_name = ?', ('unknown',))
                gender_result = cursor.fetchone()
                gender_id = gender_result[0] if gender_result else 3  # fallback
            
            # Get race ID
            race_name = demographics.get('race', 'unknown').lower()
            cursor.execute('SELECT id FROM races WHERE race_name = ?', (race_name,))
            race_result = cursor.fetchone()
            race_id = race_result[0] if race_result else None
            
            # Get emotion ID
            emotion_name = demographics.get('emotion', 'neutral').lower()
            cursor.execute('SELECT id FROM emotions WHERE emotion_name = ?', (emotion_name,))
            emotion_result = cursor.fetchone()
            emotion_id = emotion_result[0] if emotion_result else None
            
            # Insert demographic record
            cursor.execute('''
                INSERT INTO demographics (
                    person_id, detection_id, timestamp, age, age_group_id, 
                    gender_id, race_id, emotion_id, confidence, analysis_model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                person_id, detection_id, timestamp, age, age_group_id,
                gender_id, race_id, emotion_id, 
                demographics.get('confidence', 0.0), analysis_model
            ))
            
            demographic_id = cursor.lastrowid
            conn.commit()
            return demographic_id
            
        except Exception as e:
            logger.error(f"Error storing demographics: {e}")
            return None
    
    def store_dwell_time(self, person_id, zone_name, start_time, end_time=None, detection_id=None):
        """
        Store dwell time information
        
        Args:
            person_id (int): Person track ID
            zone_name (str): Zone name where dwell occurred
            start_time (datetime): Start time
            end_time (datetime): End time (None if still active)
            detection_id (int): Related detection ID
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            total_time = None
            is_active = end_time is None
            
            if not is_active:
                total_time = (end_time - start_time).total_seconds()
            
            cursor.execute('''
                INSERT INTO dwell_times (
                    person_id, detection_id, zone_name, start_time, 
                    end_time, total_time, is_active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (person_id, detection_id, zone_name, start_time, end_time, total_time, is_active))
            
            dwell_id = cursor.lastrowid
            conn.commit()
            return dwell_id
            
        except Exception as e:
            logger.error(f"Error storing dwell time: {e}")
            return None
    
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
    
    def get_demographic_records_paginated(self, page=1, per_page=10, search='', sort_by='timestamp', sort_order='desc', hours=24):
        """
        Get paginated demographic records using 3NF schema with proper joins
        
        Args:
            page: Page number (1-based)
            per_page: Records per page
            search: Search term for filtering
            sort_by: Column to sort by
            sort_order: 'asc' or 'desc'
            hours: Time range in hours
        
        Returns:
            dict: Records and pagination info
        """
        try:
            # Calculate time range
            from datetime import datetime, timedelta
            start_time = datetime.now() - timedelta(hours=hours)
            
            # Build base query with proper joins to lookup tables
            base_query = """
                SELECT 
                    d.id,
                    d.timestamp,
                    d.age,
                    ag.group_name as age_group,
                    g.display_name as gender,
                    g.icon_class as gender_icon,
                    r.display_name as race,
                    e.display_name as emotion,
                    e.icon_class as emotion_icon,
                    e.color_class as emotion_color,
                    d.confidence,
                    dt.total_time as dwell_time,
                    p.total_visits
                FROM demographics d
                JOIN persons p ON d.person_id = p.id
                LEFT JOIN age_groups ag ON d.age_group_id = ag.id
                JOIN genders g ON d.gender_id = g.id
                LEFT JOIN races r ON d.race_id = r.id
                LEFT JOIN emotions e ON d.emotion_id = e.id
                LEFT JOIN dwell_times dt ON d.detection_id = dt.detection_id
                WHERE d.timestamp >= ?
            """
            params = [start_time.isoformat()]
            
            # Add search filter across multiple fields
            if search:
                base_query += " AND (g.display_name LIKE ? OR e.display_name LIKE ? OR r.display_name LIKE ? OR ag.group_name LIKE ?)"
                search_term = f"%{search}%"
                params.extend([search_term, search_term, search_term, search_term])
            
            # Add sorting with proper column mapping
            column_mapping = {
                'timestamp': 'd.timestamp',
                'gender': 'g.display_name',
                'age': 'd.age',
                'emotion': 'e.display_name',
                'confidence': 'd.confidence',
                'dwell_time': 'dt.total_time'
            }
            
            sort_column = column_mapping.get(sort_by, 'd.timestamp')
            if sort_order.lower() not in ['asc', 'desc']:
                sort_order = 'desc'
            
            # Count total records
            count_query = f"SELECT COUNT(*) FROM ({base_query}) as counted"
            conn = self._get_connection()
            total_records = conn.execute(count_query, params).fetchone()[0]
            
            # Calculate pagination
            total_pages = (total_records + per_page - 1) // per_page
            offset = (page - 1) * per_page
            
            # Add sorting and pagination to main query
            main_query = base_query + f" ORDER BY {sort_column} {sort_order} LIMIT ? OFFSET ?"
            params.extend([per_page, offset])
            
            # Execute main query
            cursor = conn.execute(main_query, params)
            records = []
            
            for row in cursor.fetchall():
                record = {
                    'id': row[0],
                    'timestamp': row[1],
                    'age': row[2] if row[2] else 0,
                    'age_group': row[3] or 'Unknown',
                    'gender': row[4] or 'Unknown',
                    'gender_icon': row[5] or 'fas fa-question',
                    'race': row[6] or 'Unknown',
                    'emotion': row[7] or 'Neutral',
                    'emotion_icon': row[8] or 'fas fa-meh',
                    'emotion_color': row[9] or 'secondary',
                    'confidence': float(row[10]) if row[10] else 0.0,
                    'dwell_time': float(row[11]) if row[11] else 0.0,
                    'total_visits': row[12] if row[12] else 1
                }
                records.append(record)
            
            return {
                'records': records,
                'total': total_records,
                'pages': total_pages,
                'current_page': page,
                'per_page': per_page
            }
            
        except Exception as e:
            logger.error(f"Error getting paginated demographic records: {e}")
            return {
                'records': [],
                'total': 0,
                'pages': 0,
                'current_page': 1,
                'per_page': per_page
            }
    
    def clear_demographics_data(self):
        """
        Clear all demographic data from the database
        
        Returns:
            dict: Result with success status and message
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get count before deletion for confirmation
            cursor.execute("SELECT COUNT(*) FROM demographics")
            count_before = cursor.fetchone()[0]
            
            # Delete all demographic data
            cursor.execute("DELETE FROM demographics")
            
            # Get count after deletion
            cursor.execute("SELECT COUNT(*) FROM demographics")
            count_after = cursor.fetchone()[0]
            
            conn.commit()
            
            records_deleted = count_before - count_after
            logger.info(f"Cleared {records_deleted} demographic records from database")
            
            return {
                "success": True,
                "message": f"Successfully cleared {records_deleted} demographic records",
                "records_deleted": records_deleted
            }
            
        except Exception as e:
            logger.error(f"Error clearing demographics data: {e}")
            return {
                "success": False,
                "message": f"Failed to clear demographic data: {str(e)}",
                "records_deleted": 0
            }
    
    def populate_sample_data(self):
        """
        Populate database with realistic sample data using 3NF schema
        
        Returns:
            dict: Result with success status and details
        """
        try:
            import random
            from datetime import timedelta
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Clear existing data first (in proper order due to foreign keys)
            cursor.execute("DELETE FROM dwell_times")
            cursor.execute("DELETE FROM demographics")
            cursor.execute("DELETE FROM detections")
            cursor.execute("DELETE FROM persons")
            
            # Reset auto-increment counters to avoid UNIQUE constraint failures
            cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('persons', 'detections', 'demographics', 'dwell_times')")
            
            # Generate sample data for the last 7 days
            start_time = datetime.now() - timedelta(days=7)
            detection_records = 0
            demographic_records = 0
            person_records = 0
            
            # Get lookup table IDs for efficient insertion
            cursor.execute("SELECT id, gender_name FROM genders")
            gender_lookup = {row[1]: row[0] for row in cursor.fetchall()}
            
            cursor.execute("SELECT id, race_name FROM races")
            race_lookup = {row[1]: row[0] for row in cursor.fetchall()}
            
            cursor.execute("SELECT id, emotion_name FROM emotions")
            emotion_lookup = {row[1]: row[0] for row in cursor.fetchall()}
            
            cursor.execute("SELECT id, min_age, max_age FROM age_groups ORDER BY min_age")
            age_groups_data = cursor.fetchall()
            
            # Generate realistic data patterns
            for day in range(7):
                current_date = start_time + timedelta(days=day)
                
                # Business hours: more traffic during 9 AM - 6 PM
                for hour in range(24):
                    hour_time = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    # Calculate realistic visitor count based on hour
                    if 9 <= hour <= 18:  # Business hours
                        base_visitors = random.randint(8, 25)
                    elif 19 <= hour <= 21:  # Evening shoppers
                        base_visitors = random.randint(3, 12)
                    else:  # Off hours
                        base_visitors = random.randint(0, 5)
                    
                    # Weekend modifier
                    if current_date.weekday() >= 5:  # Weekend
                        base_visitors = int(base_visitors * 1.3)
                    
                    # Generate individual person records
                    for person_num in range(base_visitors):
                        # Random visit time within the hour
                        visit_time = hour_time + timedelta(
                            minutes=random.randint(0, 59),
                            seconds=random.randint(0, 59)
                        )
                        
                        # Create person record (let SQLite auto-increment the ID)
                        cursor.execute('''
                            INSERT INTO persons (first_detected, last_detected, total_visits)
                            VALUES (?, ?, ?)
                        ''', (visit_time, visit_time, 1))
                        person_id = cursor.lastrowid
                        person_records += 1
                        
                        # Generate realistic bounding box (simulating person detection)
                        x1 = random.randint(50, 800)
                        y1 = random.randint(50, 600)
                        x2 = x1 + random.randint(80, 200)  # Person width
                        y2 = y1 + random.randint(150, 300)  # Person height
                        confidence = random.uniform(0.7, 0.95)
                        
                        # Store detection
                        cursor.execute('''
                            INSERT INTO detections (person_id, timestamp, x1, y1, x2, y2, confidence, camera_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (person_id, visit_time, x1, y1, x2, y2, confidence, 'main_entrance'))
                        detection_id = cursor.lastrowid
                        detection_records += 1
                        
                        # Generate demographics (70% of people get analyzed)
                        if random.random() < 0.7:
                            # Age distribution realistic for retail
                            age_weights = [15, 30, 25, 18, 12]  # Weights for age groups
                            if len(age_groups_data) != len(age_weights):
                                # Ensure weights match age groups count
                                age_weights = [1] * len(age_groups_data)
                            
                            selected_age_group = random.choices(age_groups_data, weights=age_weights, k=1)[0]
                            age_group_id = selected_age_group[0]
                            min_age = selected_age_group[1]
                            max_age = selected_age_group[2]
                            age = random.randint(min_age, min(max_age, 99))
                            
                            # Gender distribution (slightly more female shoppers)
                            gender_id = random.choices(
                                [gender_lookup['male'], gender_lookup['female']], 
                                weights=[45, 55], 
                                k=1
                            )[0]
                            
                            # Race distribution (diverse but realistic)
                            race_names = ['white', 'black', 'asian', 'hispanic', 'other']
                            race_weights = [60, 15, 12, 10, 3]
                            race_name = random.choices(race_names, weights=race_weights, k=1)[0]
                            race_id = race_lookup.get(race_name, race_lookup['unknown'])
                            
                            # Emotion distribution (mostly positive in retail)
                            emotion_names = ['happy', 'neutral', 'surprised', 'focused', 'tired']
                            emotion_weights = [40, 35, 10, 10, 5]
                            emotion_name = random.choices(emotion_names, weights=emotion_weights, k=1)[0]
                            emotion_id = emotion_lookup.get(emotion_name, emotion_lookup['neutral'])
                            
                            demo_confidence = random.uniform(0.6, 0.9)
                            
                            cursor.execute('''
                                INSERT INTO demographics (
                                    person_id, detection_id, timestamp, age, age_group_id,
                                    gender_id, race_id, emotion_id, confidence, analysis_model
                                )
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ''', (person_id, detection_id, visit_time, age, age_group_id,
                                 gender_id, race_id, emotion_id, demo_confidence, 'buffalo_l'))
                            
                            demographic_records += 1
                            
                            # Generate dwell time (60% of analyzed people)
                            if random.random() < 0.6:
                                dwell_duration = random.uniform(30, 600)  # 30 seconds to 10 minutes
                                end_time = visit_time + timedelta(seconds=dwell_duration)
                                
                                cursor.execute('''
                                    INSERT INTO dwell_times (
                                        person_id, detection_id, zone_name, start_time,
                                        end_time, total_time, is_active
                                    )
                                    VALUES (?, ?, ?, ?, ?, ?, ?)
                                ''', (person_id, detection_id, 'main_area', visit_time,
                                     end_time, dwell_duration, False))
            
            conn.commit()
            
            logger.info(f"Created {person_records} person records, {detection_records} detection records, and {demographic_records} demographic records")
            
            return {
                "success": True,
                "message": f"Successfully created {person_records} persons, {detection_records} detections, and {demographic_records} demographic records",
                "person_records": person_records,
                "detection_records": detection_records,
                "demographic_records": demographic_records,
                "days_of_data": 7
            }
            
        except Exception as e:
            logger.error(f"Error populating sample data: {e}")
            return {
                "success": False,
                "message": f"Failed to populate sample data: {str(e)}",
                "person_records": 0,
                "detection_records": 0,
                "demographic_records": 0
            }
    
    def get_analytics_summary(self, hours=24):
        """
        Get analytics summary using 3NF schema with proper joins
        
        Args:
            hours (int): Number of hours to look back
            
        Returns:
            dict: Analytics summary
        """
        try:
            from datetime import timedelta
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Get total visitors (unique persons detected)
            cursor.execute('''
                SELECT COUNT(DISTINCT p.id) as unique_visitors,
                       COUNT(d.id) as total_detections
                FROM persons p
                LEFT JOIN detections d ON p.id = d.person_id
                WHERE p.first_detected >= ? AND p.first_detected <= ?
            ''', (start_time, end_time))
            
            visitor_data = cursor.fetchone()
            total_visitors = visitor_data[0] if visitor_data else 0
            total_detections = visitor_data[1] if visitor_data else 0
            
            # Get gender distribution using proper joins
            cursor.execute('''
                SELECT g.display_name, COUNT(*) as count
                FROM demographics dm
                JOIN genders g ON dm.gender_id = g.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
                GROUP BY g.id, g.display_name
            ''', (start_time, end_time))
            
            gender_data = cursor.fetchall()
            gender_summary = {row[0]: row[1] for row in gender_data}
            
            # Calculate gender ratio
            male_count = gender_summary.get('Male', 0)
            female_count = gender_summary.get('Female', 0)
            gender_ratio = f"{male_count}/{female_count}"
            
            # Get age group distribution
            cursor.execute('''
                SELECT ag.group_name, COUNT(*) as count
                FROM demographics dm
                JOIN age_groups ag ON dm.age_group_id = ag.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
                GROUP BY ag.id, ag.group_name
                ORDER BY ag.display_order
            ''', (start_time, end_time))
            
            age_data = cursor.fetchall()
            age_groups = {row[0]: row[1] for row in age_data}
            
            # Get average age
            cursor.execute('''
                SELECT AVG(dm.age) as avg_age
                FROM demographics dm
                WHERE dm.timestamp >= ? AND dm.timestamp <= ? AND dm.age IS NOT NULL
            ''', (start_time, end_time))
            
            avg_age_result = cursor.fetchone()
            avg_age = round(avg_age_result[0], 1) if avg_age_result and avg_age_result[0] else 0
            
            # Get emotion distribution
            cursor.execute('''
                SELECT e.display_name, COUNT(*) as count
                FROM demographics dm
                JOIN emotions e ON dm.emotion_id = e.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
                GROUP BY e.id, e.display_name
            ''', (start_time, end_time))
            
            emotion_data = cursor.fetchall()
            emotions = {row[0]: row[1] for row in emotion_data}
            
            # Get race distribution
            cursor.execute('''
                SELECT r.display_name, COUNT(*) as count
                FROM demographics dm
                JOIN races r ON dm.race_id = r.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
                GROUP BY r.id, r.display_name
            ''', (start_time, end_time))
            
            race_data = cursor.fetchall()
            races = {row[0]: row[1] for row in race_data}
            
            # Calculate average dwell time from dwell_times table
            cursor.execute('''
                SELECT AVG(dt.total_time) as avg_dwell_seconds
                FROM dwell_times dt
                WHERE dt.start_time >= ? AND dt.start_time <= ? 
                AND dt.total_time IS NOT NULL
            ''', (start_time, end_time))
            
            dwell_result = cursor.fetchone()
            avg_dwell_seconds = dwell_result[0] if dwell_result and dwell_result[0] else 0
            avg_dwell_time = round(avg_dwell_seconds / 60, 1) if avg_dwell_seconds > 0 else 0  # Convert to minutes
            
            # Simulate conversion rate (simplified metric based on dwell time)
            if avg_dwell_time > 0:
                # Higher dwell time suggests higher engagement/conversion
                conversion_rate = min(95, max(20, 30 + (avg_dwell_time * 2)))
            else:
                conversion_rate = 25  # Base conversion rate
            
            return {
                "success": True,
                "period_hours": hours,
                "total_visitors": total_visitors,
                "total_detections": total_detections,
                "avg_dwell_time": avg_dwell_time,
                "conversion_rate": round(conversion_rate, 1),
                "gender_ratio": gender_ratio,
                "gender_distribution": gender_summary,
                "age_groups": age_groups,
                "avg_age": avg_age,
                "emotions": emotions,
                "races": races,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting analytics summary: {e}")
            return {
                "success": False,
                "message": f"Failed to get analytics summary: {str(e)}"
            }
    
    def get_hourly_traffic(self, hours=24):
        """
        Get hourly traffic data for the specified period
        
        Args:
            hours (int): Number of hours to look back
            
        Returns:
            dict: Hourly traffic data
        """
        try:
            from datetime import timedelta
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Get hourly visitor counts
            cursor.execute('''
                SELECT 
                    strftime('%H', timestamp) as hour,
                    COUNT(DISTINCT person_id) as visitors
                FROM detections 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''', (start_time, end_time))
            
            hourly_data = cursor.fetchall()
            traffic_by_hour = {f"{int(row[0]):02d}:00": row[1] for row in hourly_data}
            
            # Fill missing hours with 0
            for hour in range(24):
                hour_key = f"{hour:02d}:00"
                if hour_key not in traffic_by_hour:
                    traffic_by_hour[hour_key] = 0
            
            # Sort by hour
            sorted_traffic = dict(sorted(traffic_by_hour.items()))
            
            return {
                "success": True,
                "period_hours": hours,
                "hourly_traffic": sorted_traffic,
                "labels": list(sorted_traffic.keys()),
                "data": list(sorted_traffic.values())
            }
            
        except Exception as e:
            logger.error(f"Error getting hourly traffic: {e}")
            return {
                "success": False,
                "message": f"Failed to get hourly traffic: {str(e)}"
            }

    
    def close(self):
        """Close database connection"""
        try:
            if self.conn:
                self.conn.close()
                logger.info("SQLite database connection closed")
        except Exception as e:
            logger.error(f"Error closing SQLite database: {e}")
    
    # User authentication methods
    def create_user(self, username, email, password_hash, full_name=None, role='viewer'):
        """
        Create a new user
        
        Args:
            username (str): Unique username
            email (str): User email
            password_hash (str): Argon2 hashed password
            full_name (str): User's full name
            role (str): User role (admin, manager, viewer)
            
        Returns:
            int: User ID if successful, None otherwise
        """
        try:
            self.cursor.execute('''
                INSERT INTO users (username, email, password_hash, full_name, role)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, full_name, role))
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation failed - duplicate username/email: {e}")
            return None
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return None
    
    def get_user_by_username(self, username):
        """Get user by username"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting user by username: {e}")
            return None
    
    def get_user_by_email(self, email):
        """Get user by email"""
        try:
            self.cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
            return self.cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        try:
            self.cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            return self.cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    def update_user_login(self, user_id, session_token=None, session_expires_at=None):
        """Update user's last login time and session info"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP, 
                    failed_login_attempts = 0,
                    locked_until = NULL,
                    session_token = ?,
                    session_expires_at = ?
                WHERE id = ?
            ''', (session_token, session_expires_at, user_id))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating user login: {e}")
            return False
    
    def update_failed_login_attempts(self, username, lock_until=None):
        """Increment failed login attempts and optionally lock account"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users 
                SET failed_login_attempts = failed_login_attempts + 1,
                    locked_until = ?
                WHERE username = ?
            ''', (lock_until, username))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating failed login attempts: {e}")
            return False
    
    def log_login_attempt(self, username, ip_address, user_agent, success, failure_reason=None):
        """Log a login attempt"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO login_attempts (username, ip_address, user_agent, success, failure_reason)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, ip_address, user_agent, success, failure_reason))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error logging login attempt: {e}")
            return False
    
    def create_user_session(self, user_id, session_token, ip_address, user_agent, expires_at):
        """Create a new user session"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, ip_address, user_agent, expires_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, session_token, ip_address, user_agent, expires_at))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating user session: {e}")
            return False
    
    def get_session(self, session_token):
        """Get session by token"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT s.*, u.username, u.role, u.is_active 
                FROM user_sessions s 
                JOIN users u ON s.user_id = u.id 
                WHERE s.session_token = ? AND s.is_active = 1 AND s.expires_at > CURRENT_TIMESTAMP
            ''', (session_token,))
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return None
    
    def invalidate_session(self, session_token):
        """Invalidate a session"""
        try:
            self.cursor.execute('''
                UPDATE user_sessions SET is_active = 0 WHERE session_token = ?
            ''', (session_token,))
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error invalidating session: {e}")
            return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            self.cursor.execute('''
                UPDATE user_sessions SET is_active = 0 WHERE expires_at <= CURRENT_TIMESTAMP
            ''')
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error cleaning up expired sessions: {e}")
            return False

    def _create_default_users(self):
        """Create default users for RBAC demonstration"""
        try:
            from argon2 import PasswordHasher

            # Initialize password hasher
            ph = PasswordHasher(
                time_cost=3,
                memory_cost=65536,
                parallelism=1,
                hash_len=32,
                salt_len=16
            )

            # Define default users for each role
            default_users = [
                {
                    'username': 'admin',
                    'email': 'admin@reviision.com',
                    'password': 'admin',
                    'full_name': 'Administrator',
                    'role': 'admin'
                },
                {
                    'username': 'manager',
                    'email': 'manager@reviision.com',
                    'password': 'manager',
                    'full_name': 'Manager User',
                    'role': 'manager'
                },
                {
                    'username': 'manager2',
                    'email': 'manager2@reviision.com',
                    'password': 'manager2',
                    'full_name': 'Manager User 2',
                    'role': 'manager'
                },
                {
                    'username': 'viewer',
                    'email': 'viewer@reviision.com',
                    'password': 'viewer',
                    'full_name': 'Viewer User',
                    'role': 'viewer'
                },
                {
                    'username': 'viewer2',
                    'email': 'viewer2@reviision.com',
                    'password': 'viewer2',
                    'full_name': 'Viewer User 2',
                    'role': 'viewer'
                }
            ]

            # Create each default user if they don't exist
            for user_data in default_users:
                self.cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', (user_data['username'],))
                if self.cursor.fetchone()[0] == 0:
                    try:
                        # Hash the password
                        password_hash = ph.hash(user_data['password'])

                        # Create the user
                        self.cursor.execute('''
                            INSERT INTO users (username, email, password_hash, full_name, role, is_active)
                            VALUES (?, ?, ?, ?, ?, 1)
                        ''', (
                            user_data['username'],
                            user_data['email'],
                            password_hash,
                            user_data['full_name'],
                            user_data['role']
                        ))

                        logger.info(f"Created default user: {user_data['username']} ({user_data['role']})")

                    except Exception as e:
                        logger.error(f"Error creating default user {user_data['username']}: {e}")
                        continue
                else:
                    logger.debug(f"Default user {user_data['username']} already exists")

            # Log summary of created users
            logger.info("Default users created for RBAC demonstration:")
            logger.info("- Username: admin, Password: admin, Role: admin")
            logger.info("- Username: manager, Password: manager, Role: manager")
            logger.info("- Username: manager2, Password: manager2, Role: manager")
            logger.info("- Username: viewer, Password: viewer, Role: viewer")
            logger.info("- Username: viewer2, Password: viewer2, Role: viewer")

        except ImportError:
            logger.error("Argon2 not available, skipping default user creation")
        except Exception as e:
            logger.error(f"Error creating default users: {e}")
            # Don't raise here to avoid breaking database initialization

    def get_demographics_data(self, page=1, per_page=10, search='', sort_by='timestamp', sort_order='desc', hours=24):
        """
        Get detailed demographics data for table display
        
        Args:
            page (int): Page number
            per_page (int): Items per page
            search (str): Search query
            sort_by (str): Sort field
            sort_order (str): Sort order
            hours (int): Number of hours to look back
            
        Returns:
            dict: Demographics data for table
        """
        try:
            from datetime import timedelta
            
            conn = self._get_connection()
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Build the base query with proper joins to lookup tables
            base_query = '''
                SELECT 
                    d.id,
                    d.person_id,
                    d.timestamp,
                    d.age,
                    ag.group_name as age_group,
                    g.display_name as gender,
                    r.display_name as race,
                    e.display_name as emotion,
                    d.confidence,
                    dt.total_time as dwell_time,
                    p.first_detected,
                    p.last_detected
                FROM demographics d
                JOIN persons p ON d.person_id = p.id
                LEFT JOIN age_groups ag ON d.age_group_id = ag.id
                JOIN genders g ON d.gender_id = g.id
                LEFT JOIN races r ON d.race_id = r.id
                LEFT JOIN emotions e ON d.emotion_id = e.id
                LEFT JOIN dwell_times dt ON d.detection_id = dt.detection_id
                WHERE d.timestamp >= ? AND d.timestamp <= ?
            '''
            
            # Add search filter if provided
            params = [start_time, end_time]
            if search:
                base_query += ' AND (g.display_name LIKE ? OR r.display_name LIKE ? OR e.display_name LIKE ? OR ag.group_name LIKE ?)'
                search_term = f'%{search}%'
                params.extend([search_term, search_term, search_term, search_term])
            
            # Add sorting with proper column mapping
            column_mapping = {
                'timestamp': 'd.timestamp',
                'gender': 'g.display_name',
                'age': 'd.age',
                'emotion': 'e.display_name',
                'confidence': 'd.confidence',
                'dwell_time': 'dt.total_time'
            }
            
            sort_column = column_mapping.get(sort_by, 'd.timestamp')
            if sort_order.lower() not in ['asc', 'desc']:
                sort_order = 'desc'
            
            base_query += f' ORDER BY {sort_column} {sort_order.upper()}'
            
            # Get total count for pagination
            count_query = '''
                SELECT COUNT(*)
                FROM demographics d
                JOIN persons p ON d.person_id = p.id
                LEFT JOIN age_groups ag ON d.age_group_id = ag.id
                JOIN genders g ON d.gender_id = g.id
                LEFT JOIN races r ON d.race_id = r.id
                LEFT JOIN emotions e ON d.emotion_id = e.id
                WHERE d.timestamp >= ? AND d.timestamp <= ?
            '''
            
            count_params = [start_time, end_time]
            if search:
                count_query += ' AND (g.display_name LIKE ? OR r.display_name LIKE ? OR e.display_name LIKE ? OR ag.group_name LIKE ?)'
                count_params.extend([search_term, search_term, search_term, search_term])
            
            cursor = conn.execute(count_query, count_params)
            total_count = cursor.fetchone()[0]
            
            # Calculate pagination
            offset = (page - 1) * per_page
            base_query += f' LIMIT {per_page} OFFSET {offset}'
            
            # Execute main query
            cursor = conn.execute(base_query, params)
            rows = cursor.fetchall()
            
            # Format records
            records = []
            for row in rows:
                records.append({
                    'id': row[0],
                    'person_id': row[1],
                    'timestamp': row[2],
                    'age': row[3] if row[3] else 0,
                    'age_group': row[4] or 'Unknown',
                    'gender': row[5] or 'Unknown',
                    'race': row[6] or 'Unknown',
                    'emotion': row[7] or 'Neutral',
                    'confidence': float(row[8]) if row[8] else 0.0,
                    'dwell_time': float(row[9]) if row[9] else 0.0,
                    'first_detected': row[10],
                    'last_detected': row[11]
                })
            
            # Calculate pagination info
            total_pages = (total_count + per_page - 1) // per_page
            
            return {
                'data': records,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total_records': total_count,
                    'total_pages': total_pages,
                    'has_prev': page > 1,
                    'has_next': page < total_pages,
                    'prev_num': page - 1 if page > 1 else None,
                    'next_num': page + 1 if page < total_pages else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting demographics data: {e}")
            return {
                'data': [],
                'pagination': {
                    'page': 1,
                    'per_page': per_page,
                    'total_records': 0,
                    'total_pages': 0,
                    'has_prev': False,
                    'has_next': False,
                    'prev_num': None,
                    'next_num': None
                }
            }
    
    def get_demographic_trends(self, hours=24):
        """
        Get demographic trends over time periods
        
        Args:
            hours (int): Number of hours to look back
            
        Returns:
            dict: Demographic trends data grouped by time periods
        """
        try:
            from datetime import timedelta
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Determine time grouping based on hours range
            if hours <= 24:
                # Group by hour for daily view
                time_format = '%Y-%m-%d %H:00:00'
                time_label = 'Hour'
            elif hours <= 168:  # 7 days
                # Group by day for weekly view
                time_format = '%Y-%m-%d'
                time_label = 'Day'
            else:
                # Group by week for monthly view
                time_format = '%Y-%W'
                time_label = 'Week'
            
            # Get gender trends over time
            cursor.execute('''
                SELECT 
                    strftime(?, dm.timestamp) as time_period,
                    g.display_name as gender,
                    COUNT(*) as count
                FROM demographics dm
                JOIN genders g ON dm.gender_id = g.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
                GROUP BY time_period, g.id, g.display_name
                ORDER BY time_period, g.display_name
            ''', (time_format, start_time, end_time))
            
            gender_trends = cursor.fetchall()
            
            # Get age group trends over time
            cursor.execute('''
                SELECT 
                    strftime(?, dm.timestamp) as time_period,
                    ag.group_name as age_group,
                    COUNT(*) as count
                FROM demographics dm
                JOIN age_groups ag ON dm.age_group_id = ag.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
                GROUP BY time_period, ag.id, ag.group_name
                ORDER BY time_period, ag.display_order
            ''', (time_format, start_time, end_time))
            
            age_trends = cursor.fetchall()
            
            # Get emotion trends over time
            cursor.execute('''
                SELECT 
                    strftime(?, dm.timestamp) as time_period,
                    e.display_name as emotion,
                    COUNT(*) as count
                FROM demographics dm
                JOIN emotions e ON dm.emotion_id = e.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
                GROUP BY time_period, e.id, e.display_name
                ORDER BY time_period, e.display_name
            ''', (time_format, start_time, end_time))
            
            emotion_trends = cursor.fetchall()
            
            # Process gender trends
            gender_data = {}
            time_periods = set()
            
            for row in gender_trends:
                time_period, gender, count = row
                time_periods.add(time_period)
                
                if gender not in gender_data:
                    gender_data[gender] = {}
                gender_data[gender][time_period] = count
            
            # Process age trends
            age_data = {}
            for row in age_trends:
                time_period, age_group, count = row
                time_periods.add(time_period)
                
                if age_group not in age_data:
                    age_data[age_group] = {}
                age_data[age_group][time_period] = count
            
            # Process emotion trends
            emotion_data = {}
            for row in emotion_trends:
                time_period, emotion, count = row
                time_periods.add(time_period)
                
                if emotion not in emotion_data:
                    emotion_data[emotion] = {}
                emotion_data[emotion][time_period] = count
            
            # Convert to sorted list
            sorted_periods = sorted(list(time_periods))
            
            # Fill in missing periods with zeros
            for gender in gender_data:
                for period in sorted_periods:
                    if period not in gender_data[gender]:
                        gender_data[gender][period] = 0
            
            for age_group in age_data:
                for period in sorted_periods:
                    if period not in age_data[age_group]:
                        age_data[age_group][period] = 0
            
            for emotion in emotion_data:
                for period in sorted_periods:
                    if period not in emotion_data[emotion]:
                        emotion_data[emotion][period] = 0
            
            return {
                "success": True,
                "time_periods": sorted_periods,
                "time_label": time_label,
                "gender_trends": gender_data,
                "age_trends": age_data,
                "emotion_trends": emotion_data,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "hours": hours
            }
            
        except Exception as e:
            logger.error(f"Error getting demographic trends: {e}")
            return {
                "success": False,
                "message": f"Failed to get demographic trends: {str(e)}",
                "time_periods": [],
                "time_label": "Time",
                "gender_trends": {},
                "age_trends": {},
                "emotion_trends": {}
            }
