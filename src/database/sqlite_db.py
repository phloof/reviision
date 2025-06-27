"""
SQLite Database implementation for Retail Analytics System
"""

import os
import sqlite3
import logging
import threading
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
        """Initialize database tables"""
        try:
            # Use the main connection for initialization
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=20.0
            )
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
            
            self.conn.commit()
            logger.info(f"SQLite database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")
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