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
import time

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
        self._lock = threading.RLock()  # Use RLock for nested locking
        self._connection_pool = {}  # Simple connection pool
        self._pool_lock = threading.Lock()

        # Initialize database
        self._init_db()
        
    def _get_connection(self):
        """Get a thread-safe database connection with retry mechanism"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            # For in-memory databases, reuse the main connection so that schema persists
            if self.db_path == ':memory:':
                self._local.conn = self.conn
            else:
                max_retries = 3
                retry_count = 0
                while retry_count < max_retries:
                    try:
                        self._local.conn = sqlite3.connect(
                            self.db_path,
                            check_same_thread=False,
                            timeout=30.0  # Increased timeout
                        )
                        break
                    except sqlite3.OperationalError as e:
                        if "database is locked" in str(e).lower() and retry_count < max_retries - 1:
                            retry_count += 1
                            logger.warning(f"Database locked, retrying ({retry_count}/{max_retries})...")
                            time.sleep(0.1 * retry_count)  # Exponential backoff
                            continue
                        else:
                            logger.error(f"Failed to connect to database after {max_retries} retries: {e}")
                            raise
                    except Exception as e:
                        logger.error(f"Unexpected database connection error: {e}")
                        raise
                        
            self._local.conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._local.conn.execute('PRAGMA journal_mode=WAL')
            self._local.conn.execute('PRAGMA synchronous=NORMAL')
            self._local.conn.execute('PRAGMA cache_size=10000')
            self._local.conn.execute('PRAGMA temp_store=memory')
            self._local.conn.execute('PRAGMA busy_timeout=30000')  # 30 second busy timeout
        return self._local.conn
    
    def _execute_with_retry(self, operation_func, *args, **kwargs):
        """Execute database operation with retry mechanism for handling locks"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                return operation_func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and retry_count < max_retries - 1:
                    retry_count += 1
                    logger.warning(f"Database operation failed (locked), retrying ({retry_count}/{max_retries})...")
                    time.sleep(0.1 * retry_count)  # Exponential backoff
                    continue
                else:
                    logger.error(f"Database operation failed after {max_retries} retries: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected database operation error: {e}")
                raise
    
    def _init_db(self):
        """Initialize database tables with 3NF-compliant schema"""
        try:
            # Use the main connection for initialization
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    self.conn = sqlite3.connect(
                        self.db_path,
                        check_same_thread=False,
                        timeout=30.0
                    )
                    break
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e).lower() and retry_count < max_retries - 1:
                        retry_count += 1
                        logger.warning(f"Database locked during initialization, retrying ({retry_count}/{max_retries})...")
                        time.sleep(0.5 * retry_count)
                        continue
                    else:
                        logger.error(f"Failed to initialize database after {max_retries} retries: {e}")
                        raise
                        
            self.conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key constraints
            self.cursor = self.conn.cursor()
            
            logger.info(f"Initializing SQLite database at: {self.db_path}")
            
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
            
            # Zones table (new) ----------------------------------------------------
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS zones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT DEFAULT 'default',
                    name TEXT NOT NULL UNIQUE,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL,
                    y2 INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_zones_camera ON zones(camera_id)')
            
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
            
            # Dwell times table (zone-aware) --------------------------------------
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS dwell_times (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    zone_id INTEGER NOT NULL DEFAULT 0,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME,
                    total_time REAL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
                    FOREIGN KEY (zone_id) REFERENCES zones(id)
                )
            ''')
            # Ensure legacy databases are upgraded to include zone_id column
            try:
                cursor = self.conn.cursor()
                cursor.execute("PRAGMA table_info(dwell_times)")
                cols = [row[1] for row in cursor.fetchall()]
                if 'zone_id' not in cols:
                    logger.info("Upgrading existing dwell_times table with zone_id column")
                    cursor.execute('ALTER TABLE dwell_times ADD COLUMN zone_id INTEGER NOT NULL DEFAULT 0')
                if 'zone_name' in cols:
                    # SQLite cannot drop columns before v3.35; skip if unsupported
                    logger.info("Legacy column zone_name present – retained for backward compatibility")
                self.conn.commit()
            except Exception as upgrade_err:
                logger.warning(f"Dwell_times upgrade check failed: {upgrade_err}")
            
            # Create index now that column is guaranteed
            self.cursor.execute('CREATE INDEX IF NOT EXISTS idx_dwell_times_zone ON dwell_times(zone_id)')
            
            # Customer paths table for path tracking and analysis
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS customer_paths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    session_id VARCHAR(50) NOT NULL,
                    sequence_number INTEGER NOT NULL,
                    x_position INTEGER NOT NULL,
                    y_position INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    movement_type VARCHAR(20) DEFAULT 'walking',
                    speed REAL DEFAULT 0.0,
                    direction_angle REAL,
                    zone_name VARCHAR(50),
                    path_complexity REAL DEFAULT 0.0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                )
            ''')
            
            # Path segments table for storing simplified path segments
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS path_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    session_id VARCHAR(50) NOT NULL,
                    start_time DATETIME NOT NULL,
                    end_time DATETIME NOT NULL,
                    start_x INTEGER NOT NULL,
                    start_y INTEGER NOT NULL,
                    end_x INTEGER NOT NULL,
                    end_y INTEGER NOT NULL,
                    total_distance REAL NOT NULL,
                    avg_speed REAL NOT NULL,
                    segment_type VARCHAR(20) DEFAULT 'linear',
                    point_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                )
            ''')
            
            # Correlation cache table for storing analysis results
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS correlation_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_type VARCHAR(50) NOT NULL,
                    analysis_key VARCHAR(255) NOT NULL,
                    parameters_hash VARCHAR(64) NOT NULL,
                    result_data TEXT NOT NULL,
                    confidence_level REAL,
                    p_value REAL,
                    sample_size INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME NOT NULL,
                    UNIQUE(analysis_type, analysis_key, parameters_hash)
                )
            ''')
            
            # Face snapshots table for storing best quality face images per person
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER NOT NULL,
                    face_image_path VARCHAR(255),
                    face_image_data BLOB,
                    quality_score REAL NOT NULL CHECK (quality_score >= 0 AND quality_score <= 1),
                    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                    bbox_x1 INTEGER NOT NULL,
                    bbox_y1 INTEGER NOT NULL,
                    bbox_x2 INTEGER NOT NULL,
                    bbox_y2 INTEGER NOT NULL,
                    face_width INTEGER GENERATED ALWAYS AS (bbox_x2 - bbox_x1) VIRTUAL,
                    face_height INTEGER GENERATED ALWAYS AS (bbox_y2 - bbox_y1) VIRTUAL,
                    is_primary BOOLEAN DEFAULT FALSE,
                    embedding_vector BLOB, -- Store face embedding for ReID
                    analysis_method VARCHAR(50),
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                )
            ''')
            
            # Create indexes for face snapshots for performance
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_face_snapshots_person_primary 
                ON face_snapshots(person_id, is_primary)
            ''')
            
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_face_snapshots_quality 
                ON face_snapshots(person_id, quality_score DESC)
            ''')
            
            self.cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_face_snapshots_timestamp 
                ON face_snapshots(timestamp DESC)
            ''')
            
            # Trigger to ensure only one primary face per person
            self.cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS ensure_single_primary_face
                AFTER INSERT ON face_snapshots
                WHEN NEW.is_primary = 1
                BEGIN
                    UPDATE face_snapshots 
                    SET is_primary = 0 
                    WHERE person_id = NEW.person_id 
                    AND id != NEW.id 
                    AND is_primary = 1;
                END
            ''')
            
            self.cursor.execute('''
                CREATE TRIGGER IF NOT EXISTS ensure_single_primary_face_update
                AFTER UPDATE ON face_snapshots
                WHEN NEW.is_primary = 1 AND OLD.is_primary != 1
                BEGIN
                    UPDATE face_snapshots 
                    SET is_primary = 0 
                    WHERE person_id = NEW.person_id 
                    AND id != NEW.id 
                    AND is_primary = 1;
                END
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
                'CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)',
                
                # Customer paths indexes for performance
                'CREATE INDEX IF NOT EXISTS idx_paths_person_session ON customer_paths(person_id, session_id)',
                'CREATE INDEX IF NOT EXISTS idx_paths_timestamp ON customer_paths(timestamp)',
                'CREATE INDEX IF NOT EXISTS idx_paths_sequence ON customer_paths(session_id, sequence_number)',
                'CREATE INDEX IF NOT EXISTS idx_paths_zone ON customer_paths(zone_name)',
                'CREATE INDEX IF NOT EXISTS idx_paths_movement_type ON customer_paths(movement_type)',
                
                # Path segments indexes
                'CREATE INDEX IF NOT EXISTS idx_segments_person ON path_segments(person_id)',
                'CREATE INDEX IF NOT EXISTS idx_segments_session ON path_segments(session_id)',
                'CREATE INDEX IF NOT EXISTS idx_segments_time ON path_segments(start_time, end_time)',
                'CREATE INDEX IF NOT EXISTS idx_segments_type ON path_segments(segment_type)',
                
                # Correlation cache indexes
                'CREATE INDEX IF NOT EXISTS idx_correlation_type_key ON correlation_cache(analysis_type, analysis_key)',
                'CREATE INDEX IF NOT EXISTS idx_correlation_expires ON correlation_cache(expires_at)',
                'CREATE INDEX IF NOT EXISTS idx_correlation_created ON correlation_cache(created_at)'
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
        def _detection_operation():
            if timestamp is None:
                operation_timestamp = datetime.now()
            else:
                operation_timestamp = timestamp
                
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Ensure person exists in persons table
            cursor.execute('SELECT id FROM persons WHERE id = ?', (person_id,))
            if not cursor.fetchone():
                # Create new person record
                cursor.execute('''
                    INSERT INTO persons (id, first_detected, last_detected, total_visits)
                    VALUES (?, ?, ?, 1)
                ''', (person_id, operation_timestamp, operation_timestamp))
            else:
                # Update last detected time and increment visits
                cursor.execute('''
                    UPDATE persons 
                    SET last_detected = ?, total_visits = total_visits + 1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (operation_timestamp, person_id))
            
            # Insert detection record
            cursor.execute('''
                INSERT INTO detections (person_id, timestamp, x1, y1, x2, y2, confidence, camera_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (person_id, operation_timestamp, bbox[0], bbox[1], bbox[2], bbox[3], confidence, camera_id))
            
            detection_id = cursor.lastrowid
            conn.commit()
            return detection_id
        
        try:
            return self._execute_with_retry(_detection_operation)
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
        def _store_operation():
            if timestamp is None:
                operation_timestamp = datetime.now()
            else:
                operation_timestamp = timestamp
                
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
            gender_name = demographics.get('gender', 'unknown').lower().strip()
            
            # Normalize gender names to match database entries
            gender_mapping = {
                'male': 'male',
                'man': 'male', 
                'female': 'female',
                'woman': 'female',
                'unknown': 'unknown',
                'analyzing...': 'unknown'
            }
            
            normalized_gender = gender_mapping.get(gender_name, 'unknown')
            logger.debug(f"Gender mapping: '{gender_name}' -> '{normalized_gender}'")
            
            cursor.execute('SELECT id FROM genders WHERE gender_name = ?', (normalized_gender,))
            gender_result = cursor.fetchone()
            gender_id = gender_result[0] if gender_result else None
            
            # If gender not found, default to unknown
            if gender_id is None:
                cursor.execute('SELECT id FROM genders WHERE gender_name = ?', ('unknown',))
                gender_result = cursor.fetchone()
                gender_id = gender_result[0] if gender_result else 3  # fallback
                logger.warning(f"Gender '{normalized_gender}' not found in database, using unknown (ID: {gender_id})")
            
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
                person_id, detection_id, operation_timestamp, age, age_group_id,
                gender_id, race_id, emotion_id, 
                demographics.get('confidence', 0.0), analysis_model
            ))
            
            demographic_id = cursor.lastrowid
            conn.commit()
            return demographic_id
        
        try:
            return self._execute_with_retry(_store_operation)
        except Exception as e:
            logger.error(f"Error storing demographics: {e}")
            return None
    
    def store_dwell_time(self, person_id, zone_id, start_time, end_time=None):
        """
        Store dwell time information
        
        Args:
            person_id (int): Person track ID
            zone_id (int): Zone identifier (0 = outside zones)
            start_time (datetime): Start time
            end_time (datetime): End time (None if still active)
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
                    person_id, zone_id, start_time, 
                    end_time, total_time, is_active
                )
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (person_id, zone_id, start_time, end_time, total_time, is_active))
            
            dwell_id = cursor.lastrowid
            conn.commit()
            return dwell_id
            
        except Exception as e:
            logger.error(f"Error storing dwell time: {e}")
            return None
    
    def store_face_snapshot(self, person_id, face_image_data, quality_score, confidence, 
                           bbox, embedding_vector=None, analysis_method=None, 
                           timestamp=None, is_primary=False, face_image_path=None):
        """
        Store a face snapshot for a person
        
        Args:
            person_id (int): Person track ID
            face_image_data (bytes): Face image as binary data
            quality_score (float): Quality score (0-1)
            confidence (float): Detection confidence (0-1)
            bbox (tuple): Bounding box (x1, y1, x2, y2)
            embedding_vector (bytes): Serialized face embedding
            analysis_method (str): Method used for analysis
            timestamp (datetime): When snapshot was taken
            is_primary (bool): Whether this is the primary face for the person
            face_image_path (str): Optional file path if storing on disk
            
        Returns:
            int: Face snapshot ID or None if failed
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Ensure person exists
            cursor.execute('SELECT id FROM persons WHERE id = ?', (person_id,))
            if not cursor.fetchone():
                logger.warning(f"Person {person_id} not found, cannot store face snapshot")
                return None
            
            cursor.execute('''
                INSERT INTO face_snapshots (
                    person_id, face_image_data, face_image_path, quality_score, 
                    confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, 
                    embedding_vector, analysis_method, timestamp, is_primary
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                person_id, face_image_data, face_image_path, quality_score, 
                confidence, bbox[0], bbox[1], bbox[2], bbox[3],
                embedding_vector, analysis_method, timestamp, is_primary
            ))
            
            snapshot_id = cursor.lastrowid
            conn.commit()
            
            logger.info(f"Stored face snapshot {snapshot_id} for person {person_id} (quality: {quality_score:.3f})")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Error storing face snapshot: {e}")
            return None
    
    def get_primary_face_snapshot(self, person_id):
        """
        Get the primary face snapshot for a person
        
        Args:
            person_id (int): Person track ID
            
        Returns:
            dict: Face snapshot record or None
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM face_snapshots 
                WHERE person_id = ? AND is_primary = 1
                ORDER BY quality_score DESC, timestamp DESC
                LIMIT 1
            ''', (person_id,))
            
            result = cursor.fetchone()
            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            return None
            
        except Exception as e:
            logger.error(f"Error getting primary face snapshot: {e}")
            return None
    
    def get_best_face_snapshot(self, person_id):
        """
        Get the best quality face snapshot for a person (may not be primary)
        
        Args:
            person_id (int): Person track ID
            
        Returns:
            dict: Best face snapshot record or None
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM face_snapshots 
                WHERE person_id = ?
                ORDER BY quality_score DESC, confidence DESC, timestamp DESC
                LIMIT 1
            ''', (person_id,))
            
            result = cursor.fetchone()
            if result:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, result))
            return None
            
        except Exception as e:
            logger.error(f"Error getting best face snapshot: {e}")
            return None
    
    def update_primary_face_snapshot(self, person_id, snapshot_id):
        """
        Set a specific face snapshot as primary for a person
        
        Args:
            person_id (int): Person track ID
            snapshot_id (int): Face snapshot ID to make primary
            
        Returns:
            bool: Success status
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # First verify the snapshot belongs to the person
            cursor.execute('''
                SELECT id FROM face_snapshots 
                WHERE id = ? AND person_id = ?
            ''', (snapshot_id, person_id))
            
            if not cursor.fetchone():
                logger.warning(f"Face snapshot {snapshot_id} not found for person {person_id}")
                return False
            
            # Update the snapshot to be primary (trigger will handle making others non-primary)
            cursor.execute('''
                UPDATE face_snapshots 
                SET is_primary = 1 
                WHERE id = ?
            ''', (snapshot_id,))
            
            conn.commit()
            logger.info(f"Set face snapshot {snapshot_id} as primary for person {person_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating primary face snapshot: {e}")
            return False
    
    def cleanup_old_face_snapshots(self, person_id, max_snapshots=3):
        """
        Clean up old face snapshots, keeping only the best quality ones
        
        Args:
            person_id (int): Person track ID
            max_snapshots (int): Maximum snapshots to keep per person
            
        Returns:
            int: Number of snapshots removed
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get all snapshots for the person, ordered by quality
            cursor.execute('''
                SELECT id, is_primary FROM face_snapshots 
                WHERE person_id = ?
                ORDER BY quality_score DESC, confidence DESC, timestamp DESC
            ''', (person_id,))
            
            snapshots = cursor.fetchall()
            
            if len(snapshots) <= max_snapshots:
                return 0  # No cleanup needed
            
            # Keep the best ones and primary, remove the rest
            snapshots_to_keep = []
            primary_found = False
            
            for snapshot_id, is_primary in snapshots:
                if is_primary:
                    snapshots_to_keep.append(snapshot_id)
                    primary_found = True
                elif len(snapshots_to_keep) < max_snapshots:
                    snapshots_to_keep.append(snapshot_id)
            
            # If we kept more than max because of primary, remove some non-primary ones
            if len(snapshots_to_keep) > max_snapshots and primary_found:
                # Keep primary + (max_snapshots - 1) best quality
                keep_primary = [sid for sid, is_primary in snapshots if is_primary][0]
                keep_others = [sid for sid, is_primary in snapshots if not is_primary][:max_snapshots-1]
                snapshots_to_keep = [keep_primary] + keep_others
            
            # Remove the excess snapshots
            snapshots_to_remove = [sid for sid, _ in snapshots if sid not in snapshots_to_keep]
            
            if snapshots_to_remove:
                cursor.executemany(
                    'DELETE FROM face_snapshots WHERE id = ?',
                    [(sid,) for sid in snapshots_to_remove]
                )
                conn.commit()
                
                logger.info(f"Cleaned up {len(snapshots_to_remove)} old face snapshots for person {person_id}")
                return len(snapshots_to_remove)
            
            return 0
            
        except Exception as e:
            logger.error(f"Error cleaning up face snapshots: {e}")
            return 0
    
    def has_good_demographics(self, person_id, min_confidence=0.7):
        """
        Check if a person already has good quality demographic data
        
        Args:
            person_id (int): Person track ID
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            bool: True if person has good demographics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT confidence FROM demographics 
                WHERE person_id = ? AND confidence >= ?
                ORDER BY confidence DESC, timestamp DESC
                LIMIT 1
            ''', (person_id, min_confidence))
            
            result = cursor.fetchone()
            return result is not None
            
        except Exception as e:
            logger.error(f"Error checking demographics quality: {e}")
            return False
    
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
    
    def get_total_persons_count(self):
        """
        Get total count of persons in database
        
        Returns:
            int: Total number of persons in database
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM persons')
            result = cursor.fetchone()
            return result[0] if result else 0
            
        except Exception as e:
            logger.error(f"Error getting total persons count: {e}")
            return 0

    def get_analytics_summary(self, hours=24, age_filter=None, gender_filter=None):
        """
        Get analytics summary using 3NF schema with proper joins and optional demographic filtering

        Args:
            hours (int): Number of hours to look back
            age_filter (str): Age group filter ('child', 'teen', 'adult', 'senior')
            gender_filter (str): Gender filter ('male', 'female')

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

            # Build demographic filter conditions
            demographic_filter = ""
            demographic_params = []

            if age_filter or gender_filter:
                demographic_filter = " AND EXISTS (SELECT 1 FROM demographics dm2 WHERE dm2.person_id = p.id"
                if age_filter:
                    age_group_map = {
                        'child': '0-12',
                        'teen': '13-19',
                        'adult': '20-64',
                        'senior': '65+'
                    }
                    if age_filter in age_group_map:
                        demographic_filter += " AND EXISTS (SELECT 1 FROM age_groups ag WHERE dm2.age_group_id = ag.id AND ag.group_name = ?)"
                        demographic_params.append(age_group_map[age_filter])

                if gender_filter:
                    demographic_filter += " AND EXISTS (SELECT 1 FROM genders g WHERE dm2.gender_id = g.id AND g.gender_name = ?)"
                    demographic_params.append(gender_filter)

                demographic_filter += ")"

            # Get total visitors (unique persons detected) with demographic filtering
            visitor_query = f'''
                SELECT COUNT(DISTINCT p.id) as unique_visitors,
                       COUNT(d.id) as total_detections
                FROM persons p
                LEFT JOIN detections d ON p.id = d.person_id
                WHERE p.first_detected >= ? AND p.first_detected <= ?
                {demographic_filter}
            '''

            cursor.execute(visitor_query, [start_time, end_time] + demographic_params)
            
            visitor_data = cursor.fetchone()
            total_visitors = visitor_data[0] if visitor_data else 0
            total_detections = visitor_data[1] if visitor_data else 0
            
            # Get gender distribution using proper joins with demographic filtering
            gender_query = '''
                SELECT g.display_name, COUNT(*) as count
                FROM demographics dm
                JOIN genders g ON dm.gender_id = g.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
            '''
            gender_params = [start_time, end_time]

            if age_filter:
                age_group_map = {
                    'child': '0-12',
                    'teen': '13-19',
                    'adult': '20-64',
                    'senior': '65+'
                }
                if age_filter in age_group_map:
                    gender_query += " AND EXISTS (SELECT 1 FROM age_groups ag WHERE dm.age_group_id = ag.id AND ag.group_name = ?)"
                    gender_params.append(age_group_map[age_filter])

            if gender_filter:
                gender_query += " AND g.gender_name = ?"
                gender_params.append(gender_filter)

            gender_query += " GROUP BY g.id, g.display_name"

            cursor.execute(gender_query, gender_params)
            
            gender_data = cursor.fetchall()
            gender_summary = {row[0]: row[1] for row in gender_data}
            
            # Calculate gender ratio
            male_count = gender_summary.get('Male', 0)
            female_count = gender_summary.get('Female', 0)
            gender_ratio = f"{male_count}/{female_count}"
            
            # Get age group distribution with demographic filtering
            age_query = '''
                SELECT ag.group_name, COUNT(*) as count
                FROM demographics dm
                JOIN age_groups ag ON dm.age_group_id = ag.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
            '''
            age_params = [start_time, end_time]

            if age_filter:
                age_group_map = {
                    'child': '0-12',
                    'teen': '13-19',
                    'adult': '20-64',
                    'senior': '65+'
                }
                if age_filter in age_group_map:
                    age_query += " AND ag.group_name = ?"
                    age_params.append(age_group_map[age_filter])

            if gender_filter:
                age_query += " AND EXISTS (SELECT 1 FROM genders g WHERE dm.gender_id = g.id AND g.gender_name = ?)"
                age_params.append(gender_filter)

            age_query += " GROUP BY ag.id, ag.group_name ORDER BY ag.display_order"

            cursor.execute(age_query, age_params)
            
            age_data = cursor.fetchall()
            age_groups = {row[0]: row[1] for row in age_data}
            
            # Get average age with demographic filtering
            avg_age_query = '''
                SELECT AVG(dm.age) as avg_age
                FROM demographics dm
                WHERE dm.timestamp >= ? AND dm.timestamp <= ? AND dm.age IS NOT NULL
            '''
            avg_age_params = [start_time, end_time]

            if age_filter:
                age_group_map = {
                    'child': '0-12',
                    'teen': '13-19',
                    'adult': '20-64',
                    'senior': '65+'
                }
                if age_filter in age_group_map:
                    avg_age_query += " AND EXISTS (SELECT 1 FROM age_groups ag WHERE dm.age_group_id = ag.id AND ag.group_name = ?)"
                    avg_age_params.append(age_group_map[age_filter])

            if gender_filter:
                avg_age_query += " AND EXISTS (SELECT 1 FROM genders g WHERE dm.gender_id = g.id AND g.gender_name = ?)"
                avg_age_params.append(gender_filter)

            cursor.execute(avg_age_query, avg_age_params)
            avg_age_result = cursor.fetchone()
            avg_age = round(avg_age_result[0], 1) if avg_age_result and avg_age_result[0] else 0

            # Get emotion distribution with demographic filtering
            emotion_query = '''
                SELECT e.display_name, COUNT(*) as count
                FROM demographics dm
                JOIN emotions e ON dm.emotion_id = e.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
            '''
            emotion_params = [start_time, end_time]

            if age_filter:
                age_group_map = {
                    'child': '0-12',
                    'teen': '13-19',
                    'adult': '20-64',
                    'senior': '65+'
                }
                if age_filter in age_group_map:
                    emotion_query += " AND EXISTS (SELECT 1 FROM age_groups ag WHERE dm.age_group_id = ag.id AND ag.group_name = ?)"
                    emotion_params.append(age_group_map[age_filter])

            if gender_filter:
                emotion_query += " AND EXISTS (SELECT 1 FROM genders g WHERE dm.gender_id = g.id AND g.gender_name = ?)"
                emotion_params.append(gender_filter)

            emotion_query += " GROUP BY e.id, e.display_name"

            cursor.execute(emotion_query, emotion_params)
            
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
                conversion_rate = 0  # No data available

            # Get peak hour based on last 30 days
            peak_hour = self.get_peak_hour_last_30_days()

            return {
                "success": True,
                "period_hours": hours,
                "total_visitors": total_visitors,
                "total_detections": total_detections,
                "avg_dwell_time": avg_dwell_time,
                "conversion_rate": round(conversion_rate, 1),
                "peak_hour": peak_hour,
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
    
    def get_hourly_traffic(self, hours=24, age_filter=None, gender_filter=None):
        """
        Get hourly traffic data for the specified period with optional demographic filtering

        Args:
            hours (int): Number of hours to look back
            age_filter (str): Age group filter ('child', 'teen', 'adult', 'senior')
            gender_filter (str): Gender filter ('male', 'female')

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
            
            # Get hourly visitor counts with demographic filtering
            traffic_query = '''
                SELECT
                    strftime('%H', d.timestamp) as hour,
                    COUNT(DISTINCT d.person_id) as visitors
                FROM detections d
                WHERE d.timestamp >= ? AND d.timestamp <= ?
            '''
            traffic_params = [start_time, end_time]

            if age_filter or gender_filter:
                traffic_query += '''
                    AND EXISTS (
                        SELECT 1 FROM demographics dm
                        WHERE dm.person_id = d.person_id
                '''

                if age_filter:
                    age_group_map = {
                        'child': '0-12',
                        'teen': '13-19',
                        'adult': '20-64',
                        'senior': '65+'
                    }
                    if age_filter in age_group_map:
                        traffic_query += " AND EXISTS (SELECT 1 FROM age_groups ag WHERE dm.age_group_id = ag.id AND ag.group_name = ?)"
                        traffic_params.append(age_group_map[age_filter])

                if gender_filter:
                    traffic_query += " AND EXISTS (SELECT 1 FROM genders g WHERE dm.gender_id = g.id AND g.gender_name = ?)"
                    traffic_params.append(gender_filter)

                traffic_query += ")"

            traffic_query += '''
                GROUP BY strftime('%H', d.timestamp)
                ORDER BY hour
            '''

            cursor.execute(traffic_query, traffic_params)
            
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

    def get_peak_hour_last_30_days(self):
        """
        Calculate peak hour based on the last 30 days of data

        Returns:
            str: Peak hour in HH:MM format
        """
        try:
            from datetime import timedelta

            conn = self._get_connection()
            cursor = conn.cursor()

            # Calculate time range for last 30 days
            end_time = datetime.now()
            start_time = end_time - timedelta(days=30)

            # Get hourly visitor counts for last 30 days
            cursor.execute('''
                SELECT strftime('%H', timestamp) as hour,
                       COUNT(DISTINCT person_id) as visitors
                FROM detections
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY strftime('%H', timestamp)
                ORDER BY visitors DESC, hour ASC
                LIMIT 1
            ''', (start_time, end_time))

            result = cursor.fetchone()
            if result and result[0] is not None:
                hour = int(result[0])
                return f"{hour:02d}:00"
            else:
                return "--:--"

        except Exception as e:
            logger.error(f"Error calculating peak hour for last 30 days: {e}")
            return "--:--"

    def get_weekly_patterns(self, hours=24):
        """
        Get visitor patterns by day of week

        Args:
            hours (int): Number of hours to look back

        Returns:
            dict: Weekly patterns data
        """
        try:
            from datetime import timedelta

            conn = self._get_connection()
            cursor = conn.cursor()

            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            # Get visitor counts by day of week
            cursor.execute('''
                SELECT
                    CASE strftime('%w', timestamp)
                        WHEN '0' THEN 'Sunday'
                        WHEN '1' THEN 'Monday'
                        WHEN '2' THEN 'Tuesday'
                        WHEN '3' THEN 'Wednesday'
                        WHEN '4' THEN 'Thursday'
                        WHEN '5' THEN 'Friday'
                        WHEN '6' THEN 'Saturday'
                    END as day_name,
                    strftime('%w', timestamp) as day_num,
                    COUNT(DISTINCT person_id) as visitors
                FROM detections
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY strftime('%w', timestamp)
                ORDER BY day_num
            ''', (start_time, end_time))

            results = cursor.fetchall()

            # Initialize all days with 0
            days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            weekly_data = {day: 0 for day in days_order}

            # Fill in actual data
            for row in results:
                if row[0]:  # day_name
                    weekly_data[row[0]] = row[2]  # visitors

            return {
                "success": True,
                "period_hours": hours,
                "labels": days_order,
                "data": [weekly_data[day] for day in days_order],
                "weekly_data": weekly_data
            }

        except Exception as e:
            logger.error(f"Error getting weekly patterns: {e}")
            return {
                "success": False,
                "message": f"Failed to get weekly patterns: {str(e)}"
            }

    def get_peak_hour_analysis_by_day(self, days=30):
        """
        Get average peak hours by day of week over specified period

        Args:
            days (int): Number of days to analyze

        Returns:
            dict: Peak hour analysis by day
        """
        try:
            from datetime import timedelta

            conn = self._get_connection()
            cursor = conn.cursor()

            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            # Get peak hour for each day occurrence
            cursor.execute('''
                WITH daily_hourly_counts AS (
                    SELECT
                        DATE(timestamp) as date,
                        strftime('%w', timestamp) as day_of_week,
                        strftime('%H', timestamp) as hour,
                        COUNT(DISTINCT person_id) as visitors
                    FROM detections
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY DATE(timestamp), strftime('%H', timestamp)
                ),
                daily_peaks AS (
                    SELECT
                        date,
                        day_of_week,
                        hour,
                        visitors,
                        ROW_NUMBER() OVER (PARTITION BY date ORDER BY visitors DESC, hour ASC) as rn
                    FROM daily_hourly_counts
                )
                SELECT
                    CASE day_of_week
                        WHEN '0' THEN 'Sunday'
                        WHEN '1' THEN 'Monday'
                        WHEN '2' THEN 'Tuesday'
                        WHEN '3' THEN 'Wednesday'
                        WHEN '4' THEN 'Thursday'
                        WHEN '5' THEN 'Friday'
                        WHEN '6' THEN 'Saturday'
                    END as day_name,
                    day_of_week,
                    CAST(hour AS INTEGER) as peak_hour,
                    AVG(CAST(hour AS INTEGER)) as avg_peak_hour,
                    AVG(visitors) as avg_visitors,
                    COUNT(*) as occurrences
                FROM daily_peaks
                WHERE rn = 1
                GROUP BY day_of_week
                ORDER BY day_of_week
            ''', (start_time, end_time))

            results = cursor.fetchall()

            # Initialize all days
            days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            peak_hours_by_day = {}
            averages_by_day = {}

            for row in results:
                if row[0]:  # day_name
                    day_name = row[0]
                    avg_peak_hour = row[3]  # avg_peak_hour
                    avg_visitors = row[4]   # avg_visitors

                    peak_hours_by_day[day_name] = f"{int(avg_peak_hour):02d}:00"
                    averages_by_day[day_name] = round(avg_visitors, 1)

            # Fill missing days with default values
            for day in days_order:
                if day not in peak_hours_by_day:
                    peak_hours_by_day[day] = "--:--"
                    averages_by_day[day] = 0

            return {
                "success": True,
                "period_days": days,
                "peak_hours_by_day": peak_hours_by_day,
                "averages_by_day": averages_by_day,
                "labels": days_order,
                "data": [averages_by_day[day] for day in days_order]
            }

        except Exception as e:
            logger.error(f"Error getting peak hour analysis by day: {e}")
            return {
                "success": False,
                "message": f"Failed to get peak hour analysis: {str(e)}"
            }

    def get_historical_demographics(self, hours=24):
        """
        Get demographic data for historical analysis

        Args:
            hours (int): Number of hours to look back

        Returns:
            dict: Historical demographics data
        """
        try:
            from datetime import timedelta

            conn = self._get_connection()
            cursor = conn.cursor()

            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            # Get age group distribution
            cursor.execute('''
                SELECT ag.group_name, COUNT(*) as count
                FROM demographics dm
                JOIN age_groups ag ON dm.age_group_id = ag.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
                GROUP BY ag.id, ag.group_name
                ORDER BY ag.display_order
            ''', (start_time, end_time))

            age_results = cursor.fetchall()
            age_groups = {}
            for row in age_results:
                age_groups[row[0]] = {
                    'display_name': row[0],  # Use group_name as display_name
                    'count': row[1]
                }

            # Get gender distribution
            cursor.execute('''
                SELECT g.gender_name, g.display_name, COUNT(*) as count
                FROM demographics dm
                JOIN genders g ON dm.gender_id = g.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
                GROUP BY g.id, g.gender_name, g.display_name
                ORDER BY g.id
            ''', (start_time, end_time))

            gender_results = cursor.fetchall()
            gender_distribution = {}
            for row in gender_results:
                gender_distribution[row[0]] = {
                    'display_name': row[1],
                    'count': row[2]
                }

            # Get emotion distribution
            cursor.execute('''
                SELECT e.emotion_name, e.display_name, COUNT(*) as count
                FROM demographics dm
                JOIN emotions e ON dm.emotion_id = e.id
                WHERE dm.timestamp >= ? AND dm.timestamp <= ?
                GROUP BY e.id, e.emotion_name, e.display_name
                ORDER BY e.id
            ''', (start_time, end_time))

            emotion_results = cursor.fetchall()
            emotions = {}
            for row in emotion_results:
                emotions[row[0]] = {
                    'display_name': row[1],
                    'count': row[2]
                }

            return {
                "success": True,
                "period_hours": hours,
                "age_groups": age_groups,
                "gender_distribution": gender_distribution,
                "emotions": emotions,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting historical demographics: {e}")
            return {
                "success": False,
                "message": f"Failed to get historical demographics: {str(e)}"
            }

    def get_historical_dwell_time_stats(self, hours=24):
        """
        Get dwell time statistics for historical analysis

        Args:
            hours (int): Number of hours to look back

        Returns:
            dict: Dwell time statistics and trends
        """
        try:
            from datetime import timedelta
            import statistics

            conn = self._get_connection()
            cursor = conn.cursor()

            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            # Get dwell time data
            cursor.execute('''
                SELECT
                    total_time,
                    strftime('%H', start_time) as hour,
                    DATE(start_time) as date
                FROM dwell_times
                WHERE start_time >= ? AND start_time <= ?
                AND total_time IS NOT NULL
                AND total_time > 0
                ORDER BY start_time
            ''', (start_time, end_time))

            results = cursor.fetchall()

            if not results:
                return {
                    "success": True,
                    "period_hours": hours,
                    "avg_dwell_time": 0,
                    "median_dwell_time": 0,
                    "total_sessions": 0,
                    "distribution": [],
                    "hourly_trends": {},
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                }

            # Calculate basic statistics
            dwell_times = [row[0] for row in results]
            avg_dwell_time = statistics.mean(dwell_times)
            median_dwell_time = statistics.median(dwell_times)
            total_sessions = len(dwell_times)

            # Create distribution buckets (in minutes)
            distribution_buckets = {
                "0-2 min": 0,
                "2-5 min": 0,
                "5-10 min": 0,
                "10-20 min": 0,
                "20+ min": 0
            }

            for dwell_time in dwell_times:
                minutes = dwell_time / 60  # Convert seconds to minutes
                if minutes < 2:
                    distribution_buckets["0-2 min"] += 1
                elif minutes < 5:
                    distribution_buckets["2-5 min"] += 1
                elif minutes < 10:
                    distribution_buckets["5-10 min"] += 1
                elif minutes < 20:
                    distribution_buckets["10-20 min"] += 1
                else:
                    distribution_buckets["20+ min"] += 1

            # Calculate hourly trends
            hourly_data = {}
            for row in results:
                hour = row[1]
                dwell_time = row[0]
                if hour not in hourly_data:
                    hourly_data[hour] = []
                hourly_data[hour].append(dwell_time)

            hourly_trends = {}
            for hour in range(24):
                hour_str = f"{hour:02d}:00"
                if f"{hour:02d}" in hourly_data:
                    hour_dwell_times = hourly_data[f"{hour:02d}"]
                    hourly_trends[hour_str] = round(statistics.mean(hour_dwell_times) / 60, 1)  # Convert to minutes
                else:
                    hourly_trends[hour_str] = 0

            return {
                "success": True,
                "period_hours": hours,
                "avg_dwell_time": round(avg_dwell_time / 60, 1),  # Convert to minutes
                "median_dwell_time": round(median_dwell_time / 60, 1),  # Convert to minutes
                "total_sessions": total_sessions,
                "distribution": {
                    "labels": list(distribution_buckets.keys()),
                    "data": list(distribution_buckets.values())
                },
                "hourly_trends": hourly_trends,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }

        except Exception as e:
            logger.error(f"Error getting historical dwell time stats: {e}")
            return {
                "success": False,
                "message": f"Failed to get dwell time statistics: {str(e)}"
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
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            return cursor.fetchone()
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
    
    # Customer Path Storage Methods
    
    def store_path_points(self, path_points):
        """
        Store multiple path points efficiently using batch insert
        
        Args:
            path_points (list): List of path point dictionaries
            
        Returns:
            bool: Success status
        """
        if not path_points:
            return True
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Prepare data for batch insert
            insert_data = []
            for point in path_points:
                insert_data.append((
                    point['person_id'],
                    point['session_id'],
                    point['sequence_number'],
                    point['x_position'],
                    point['y_position'],
                    point['timestamp'],
                    point['confidence'],
                    point.get('movement_type', 'walking'),
                    point.get('speed', 0.0),
                    point.get('direction_angle'),
                    point.get('zone_name'),
                    point.get('path_complexity', 0.0)
                ))
            
            # Batch insert
            cursor.executemany('''
                INSERT INTO customer_paths (
                    person_id, session_id, sequence_number, x_position, y_position,
                    timestamp, confidence, movement_type, speed, direction_angle,
                    zone_name, path_complexity
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', insert_data)
            
            conn.commit()
            logger.debug(f"Stored {len(path_points)} path points")
            return True
            
        except Exception as e:
            logger.error(f"Error storing path points: {e}")
            return False
    
    def store_path_segment(self, person_id, session_id, start_time, end_time, 
                          start_x, start_y, end_x, end_y, total_distance, 
                          avg_speed, segment_type='linear', point_count=0):
        """
        Store a simplified path segment
        
        Args:
            person_id (int): Person track ID
            session_id (str): Session identifier
            start_time (datetime): Segment start time
            end_time (datetime): Segment end time
            start_x, start_y (int): Starting coordinates
            end_x, end_y (int): Ending coordinates
            total_distance (float): Total distance in pixels
            avg_speed (float): Average speed in pixels/second
            segment_type (str): Type of segment (linear, curved, stationary)
            point_count (int): Number of original points in segment
            
        Returns:
            int: Segment ID or None if failed
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO path_segments (
                    person_id, session_id, start_time, end_time,
                    start_x, start_y, end_x, end_y, total_distance,
                    avg_speed, segment_type, point_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (person_id, session_id, start_time, end_time,
                  start_x, start_y, end_x, end_y, total_distance,
                  avg_speed, segment_type, point_count))
            
            segment_id = cursor.lastrowid
            conn.commit()
            return segment_id
            
        except Exception as e:
            logger.error(f"Error storing path segment: {e}")
            return None
    
    def get_person_paths(self, person_id, session_id=None, hours=24):
        """
        Get path data for a specific person
        
        Args:
            person_id (int): Person track ID
            session_id (str, optional): Specific session ID
            hours (int): Hours to look back
            
        Returns:
            list: List of path points
        """
        try:
            from datetime import timedelta
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            if session_id:
                cursor.execute('''
                    SELECT person_id, session_id, sequence_number, x_position, y_position,
                           timestamp, confidence, movement_type, speed, direction_angle,
                           zone_name, path_complexity
                    FROM customer_paths
                    WHERE person_id = ? AND session_id = ?
                    ORDER BY sequence_number
                ''', (person_id, session_id))
            else:
                cursor.execute('''
                    SELECT person_id, session_id, sequence_number, x_position, y_position,
                           timestamp, confidence, movement_type, speed, direction_angle,
                           zone_name, path_complexity
                    FROM customer_paths
                    WHERE person_id = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp, sequence_number
                ''', (person_id, start_time, end_time))
            
            paths = []
            for row in cursor.fetchall():
                paths.append({
                    'person_id': row[0],
                    'session_id': row[1],
                    'sequence_number': row[2],
                    'x': row[3],
                    'y': row[4],
                    'timestamp': row[5],
                    'confidence': row[6],
                    'movement_type': row[7],
                    'speed': row[8],
                    'direction_angle': row[9],
                    'zone_name': row[10],
                    'path_complexity': row[11]
                })
            
            return paths
            
        except Exception as e:
            logger.error(f"Error getting person paths: {e}")
            return []
    
    def get_common_paths(self, hours=24, min_similarity=0.7):
        """
        Get common path patterns using spatial clustering
        
        Args:
            hours (int): Hours to analyze
            min_similarity (float): Minimum similarity threshold
            
        Returns:
            list: List of common path patterns
        """
        try:
            from datetime import timedelta
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Get path segments for analysis
            cursor.execute('''
                SELECT session_id, start_x, start_y, end_x, end_y, 
                       total_distance, avg_speed, segment_type
                FROM path_segments
                WHERE start_time >= ? AND end_time <= ?
                ORDER BY start_time
            ''', (start_time, end_time))
            
            segments = cursor.fetchall()
            
            # Simple clustering based on start/end positions
            clusters = {}
            for segment in segments:
                session_id, start_x, start_y, end_x, end_y, distance, speed, seg_type = segment
                
                # Create a simple cluster key based on rounded coordinates
                cluster_key = f"{start_x//50}_{start_y//50}_{end_x//50}_{end_y//50}"
                
                if cluster_key not in clusters:
                    clusters[cluster_key] = []
                clusters[cluster_key].append({
                    'session_id': session_id,
                    'start_x': start_x,
                    'start_y': start_y,
                    'end_x': end_x,
                    'end_y': end_y,
                    'distance': distance,
                    'speed': speed,
                    'type': seg_type
                })
            
            # Filter clusters with sufficient size
            common_paths = []
            for cluster_key, segments in clusters.items():
                if len(segments) >= 3:  # Minimum 3 similar paths
                    # Calculate average path properties
                    avg_start_x = sum(s['start_x'] for s in segments) / len(segments)
                    avg_start_y = sum(s['start_y'] for s in segments) / len(segments)
                    avg_end_x = sum(s['end_x'] for s in segments) / len(segments)
                    avg_end_y = sum(s['end_y'] for s in segments) / len(segments)
                    avg_distance = sum(s['distance'] for s in segments) / len(segments)
                    avg_speed = sum(s['speed'] for s in segments) / len(segments)
                    
                    common_paths.append({
                        'cluster_id': cluster_key,
                        'frequency': len(segments),
                        'avg_start': (avg_start_x, avg_start_y),
                        'avg_end': (avg_end_x, avg_end_y),
                        'avg_distance': avg_distance,
                        'avg_speed': avg_speed,
                        'segments': segments
                    })
            
            # Sort by frequency
            common_paths.sort(key=lambda x: x['frequency'], reverse=True)
            
            return common_paths
            
        except Exception as e:
            logger.error(f"Error getting common paths: {e}")
            return []
    
    def cleanup_old_paths(self, retention_days=30):
        """
        Clean up old path data based on retention policy
        
        Args:
            retention_days (int): Number of days to retain data
            
        Returns:
            int: Number of records deleted
        """
        try:
            from datetime import timedelta
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Delete old path points
            cursor.execute('''
                DELETE FROM customer_paths WHERE timestamp < ?
            ''', (cutoff_date,))
            
            paths_deleted = cursor.rowcount
            
            # Delete old path segments
            cursor.execute('''
                DELETE FROM path_segments WHERE end_time < ?
            ''', (cutoff_date,))
            
            segments_deleted = cursor.rowcount
            
            conn.commit()
            
            total_deleted = paths_deleted + segments_deleted
            if total_deleted > 0:
                logger.info(f"Cleaned up {total_deleted} old path records (paths: {paths_deleted}, segments: {segments_deleted})")
            
            return total_deleted
            
        except Exception as e:
            logger.error(f"Error cleaning up old paths: {e}")
            return 0
    
    # Correlation Analysis Methods
    
    def store_correlation_result(self, analysis_type, analysis_key, parameters_hash, 
                               result_data, confidence_level=None, p_value=None, 
                               sample_size=None, cache_hours=24):
        """
        Store correlation analysis result in cache
        
        Args:
            analysis_type (str): Type of analysis
            analysis_key (str): Unique key for analysis
            parameters_hash (str): Hash of parameters
            result_data (str): JSON encoded result data
            confidence_level (float): Statistical confidence level
            p_value (float): P-value of statistical test
            sample_size (int): Sample size used
            cache_hours (int): Hours to cache result
            
        Returns:
            bool: Success status
        """
        try:
            from datetime import timedelta
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            expires_at = datetime.now() + timedelta(hours=cache_hours)
            
            cursor.execute('''
                INSERT OR REPLACE INTO correlation_cache (
                    analysis_type, analysis_key, parameters_hash, result_data,
                    confidence_level, p_value, sample_size, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (analysis_type, analysis_key, parameters_hash, result_data,
                  confidence_level, p_value, sample_size, expires_at))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error storing correlation result: {e}")
            return False
    
    def get_correlation_result(self, analysis_type, analysis_key, parameters_hash):
        """
        Get cached correlation analysis result
        
        Args:
            analysis_type (str): Type of analysis
            analysis_key (str): Unique key for analysis
            parameters_hash (str): Hash of parameters
            
        Returns:
            dict: Cached result or None if not found/expired
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT result_data, confidence_level, p_value, sample_size, created_at
                FROM correlation_cache
                WHERE analysis_type = ? AND analysis_key = ? AND parameters_hash = ?
                AND expires_at > CURRENT_TIMESTAMP
            ''', (analysis_type, analysis_key, parameters_hash))
            
            result = cursor.fetchone()
            if result:
                return {
                    'result_data': result[0],
                    'confidence_level': result[1],
                    'p_value': result[2],
                    'sample_size': result[3],
                    'created_at': result[4]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting correlation result: {e}")
            return None
    
    def cleanup_expired_correlations(self):
        """
        Clean up expired correlation cache entries
        
        Returns:
            int: Number of records deleted
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                DELETE FROM correlation_cache WHERE expires_at <= CURRENT_TIMESTAMP
            ''')
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired correlation cache entries")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired correlations: {e}")
            return 0
    
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
                    COALESCE(dt.total_time, 0) as dwell_time,
                    p.first_detected,
                    p.last_detected
                FROM demographics d
                JOIN persons p ON d.person_id = p.id
                LEFT JOIN age_groups ag ON d.age_group_id = ag.id
                JOIN genders g ON d.gender_id = g.id
                LEFT JOIN races r ON d.race_id = r.id
                LEFT JOIN emotions e ON d.emotion_id = e.id
                LEFT JOIN dwell_times dt ON d.person_id = dt.person_id
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

    # ------------------------------------------------------------------
    # Zone CRUD operations
    # ------------------------------------------------------------------
    def create_zone(self, name, x1, y1, x2, y2, camera_id='default'):
        """Create a new zone and return its ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO zones (camera_id, name, x1, y1, x2, y2)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (camera_id, name, int(x1), int(y1), int(x2), int(y2)))
        zone_id = cursor.lastrowid
        conn.commit()
        return zone_id

    def update_zone(self, zone_id, **fields):
        """Update a zone's fields (name, coords, camera_id)"""
        if not fields:
            return False
        allowed = {'name', 'x1', 'y1', 'x2', 'y2', 'camera_id'}
        set_clauses = []
        values = []
        for k, v in fields.items():
            if k in allowed:
                set_clauses.append(f"{k} = ?")
                values.append(v)
        if not set_clauses:
            return False
        values.append(zone_id)
        sql = f"UPDATE zones SET {', '.join(set_clauses)}, updated_at=CURRENT_TIMESTAMP WHERE id = ?"
        conn = self._get_connection()
        conn.execute(sql, tuple(values))
        conn.commit()
        return True

    def delete_zone(self, zone_id):
        """Delete a zone"""
        conn = self._get_connection()
        conn.execute('DELETE FROM zones WHERE id = ?', (zone_id,))
        conn.commit()
        return True

    def get_zones(self, camera_id='default'):
        """Return list of zones for the specified camera"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM zones WHERE camera_id = ? ORDER BY id', (camera_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def reset_failed_login_attempts(self, username):
        """Reset failed attempts and unlock the account"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET failed_login_attempts = 0, locked_until = NULL WHERE username = ?
            ''', (username,))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error resetting failed login attempts: {e}")
            return False
