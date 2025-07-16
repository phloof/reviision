"""
Database Migration Script for 3NF Compliance

This script migrates existing data from the old schema to the new 3NF-compliant schema.
It handles SQLite databases.
"""

import logging
import sqlite3
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class DatabaseMigration:
    """
    Handles migration from old schema to new 3NF-compliant schema
    """
    
    def __init__(self, db_instance):
        self.db = db_instance
        self.db_type = type(db_instance).__name__
        
    def migrate_to_3nf(self):
        """
        Main migration method to convert existing data to 3NF schema
        
        Returns:
            dict: Migration results
        """
        if self.db_type == 'SQLiteDatabase':
            return self._migrate_sqlite()
        else:
            return {
                'success': False,
                'message': f'Unsupported database type: {self.db_type}. Only SQLite is supported.'
            }
    
    def _migrate_sqlite(self):
        """
        Migrate SQLite database to 3NF schema
        """
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            
            logger.info("Starting SQLite database migration to 3NF schema...")
            
            # Check if old tables exist and have data
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            # Create backup tables for existing data
            if 'demographics' in existing_tables:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS demographics_old AS 
                    SELECT * FROM demographics
                ''')
                
            if 'detections' in existing_tables:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detections_old AS 
                    SELECT * FROM detections
                ''')
            
            # Check if we're already using 3NF schema
            cursor.execute("PRAGMA table_info(demographics)")
            demo_columns = [row[1] for row in cursor.fetchall()]
            
            if 'gender_id' in demo_columns:
                logger.info("Database is already using 3NF schema")
                return {
                    'success': True,
                    'message': 'Database is already using 3NF schema',
                    'migrated_records': 0
                }
            
            # Get existing data from old schema
            cursor.execute("SELECT * FROM demographics_old")
            old_demographics = cursor.fetchall()
            
            cursor.execute("SELECT * FROM detections_old")
            old_detections = cursor.fetchall()
            
            migrated_count = 0
            
            # Clear existing data from new tables
            cursor.execute("DELETE FROM demographics")
            cursor.execute("DELETE FROM detections")
            cursor.execute("DELETE FROM persons")
            
            # Get lookup table mappings
            gender_map = self._get_lookup_mapping(cursor, 'genders', 'gender_name')
            race_map = self._get_lookup_mapping(cursor, 'races', 'race_name')
            emotion_map = self._get_lookup_mapping(cursor, 'emotions', 'emotion_name')
            
            # Migrate detections data
            person_ids = set()
            for detection in old_detections:
                person_id = detection[2] if len(detection) > 2 else detection[1]  # Handle different schemas
                person_ids.add(person_id)
                
                # Create person record if not exists
                cursor.execute('SELECT id FROM persons WHERE id = ?', (person_id,))
                if not cursor.fetchone():
                    timestamp = detection[1] if len(detection) > 2 else detection[0]
                    cursor.execute('''
                        INSERT INTO persons (id, first_detected, last_detected, total_visits)
                        VALUES (?, ?, ?, 1)
                    ''', (person_id, timestamp, timestamp))
                
                # Insert detection with new schema
                if len(detection) >= 7:  # Old schema: id, timestamp, person_id, x1, y1, x2, y2, confidence
                    cursor.execute('''
                        INSERT INTO detections (person_id, timestamp, x1, y1, x2, y2, confidence, camera_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (detection[2], detection[1], detection[3], detection[4], 
                         detection[5], detection[6], detection[7], 'main_entrance'))
            
            # Migrate demographics data
            for demo in old_demographics:
                if len(demo) >= 8:  # Old schema: id, person_id, timestamp, age, gender, race, emotion, confidence
                    person_id = demo[1]
                    timestamp = demo[2]
                    age = demo[3]
                    gender_text = demo[4] if demo[4] else 'unknown'
                    race_text = demo[5] if demo[5] else 'unknown'
                    emotion_text = demo[6] if demo[6] else 'neutral'
                    confidence = demo[7] if demo[7] else 0.0
                    
                    # Map to lookup table IDs
                    gender_id = gender_map.get(gender_text.lower(), gender_map.get('unknown', 3))
                    race_id = race_map.get(race_text.lower(), race_map.get('unknown'))
                    emotion_id = emotion_map.get(emotion_text.lower(), emotion_map.get('neutral'))
                    
                    # Determine age group
                    age_group_id = self._get_age_group_id(cursor, age)
                    
                    # Insert with new schema
                    cursor.execute('''
                        INSERT INTO demographics (
                            person_id, timestamp, age, age_group_id, gender_id, 
                            race_id, emotion_id, confidence, analysis_model
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (person_id, timestamp, age, age_group_id, gender_id, 
                         race_id, emotion_id, confidence, 'legacy_migration'))
                    
                    migrated_count += 1
            
            conn.commit()
            
            logger.info(f"Successfully migrated {migrated_count} demographic records to 3NF schema")
            
            return {
                'success': True,
                'message': f'Successfully migrated {migrated_count} records to 3NF schema',
                'migrated_records': migrated_count,
                'backup_tables': ['demographics_old', 'detections_old']
            }
            
        except Exception as e:
            logger.error(f"Error during SQLite migration: {e}")
            return {
                'success': False,
                'message': f'Migration failed: {str(e)}',
                'migrated_records': 0
            }
    

    
    def _get_lookup_mapping(self, cursor, table_name, field_name):
        """
        Get mapping from text values to IDs for lookup tables
        """
        cursor.execute(f'SELECT id, {field_name} FROM {table_name}')
        return {row[1]: row[0] for row in cursor.fetchall()}
    
    def _get_age_group_id(self, cursor, age):
        """
        Get age group ID based on age value
        """
        if age is None:
            return None
            
        cursor.execute('''
            SELECT id FROM age_groups 
            WHERE ? >= min_age AND ? <= max_age
        ''', (age, age))
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def cleanup_old_tables(self):
        """
        Remove old backup tables after successful migration
        """
        if self.db_type == 'SQLiteDatabase':
            try:
                conn = self.db._get_connection()
                cursor = conn.cursor()
                
                # Drop backup tables
                cursor.execute("DROP TABLE IF EXISTS demographics_old")
                cursor.execute("DROP TABLE IF EXISTS detections_old")
                
                conn.commit()
                logger.info("Old backup tables cleaned up successfully")
                return True
                
            except Exception as e:
                logger.error(f"Error cleaning up old tables: {e}")
                return False
        
        return True
    
    def verify_migration(self):
        """
        Verify that migration was successful
        """
        try:
            if self.db_type == 'SQLiteDatabase':
                conn = self.db._get_connection()
                cursor = conn.cursor()
                
                # Check that lookup tables are populated
                cursor.execute("SELECT COUNT(*) FROM genders")
                gender_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM age_groups")
                age_group_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM emotions")
                emotion_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM races")
                race_count = cursor.fetchone()[0]
                
                # Check that data exists in main tables
                cursor.execute("SELECT COUNT(*) FROM demographics")
                demo_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM detections")
                detection_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM persons")
                person_count = cursor.fetchone()[0]
                
                # Verify foreign key relationships
                cursor.execute('''
                    SELECT COUNT(*) FROM demographics d
                    JOIN genders g ON d.gender_id = g.id
                    JOIN persons p ON d.person_id = p.id
                ''')
                fk_count = cursor.fetchone()[0]
                
                return {
                    'success': True,
                    'lookup_tables': {
                        'genders': gender_count,
                        'age_groups': age_group_count,
                        'emotions': emotion_count,
                        'races': race_count
                    },
                    'data_tables': {
                        'persons': person_count,
                        'detections': detection_count,
                        'demographics': demo_count,
                        'foreign_key_relationships': fk_count
                    },
                    'message': 'Migration verification successful'
                }
            
        except Exception as e:
            logger.error(f"Error verifying migration: {e}")
            return {
                'success': False,
                'message': f'Migration verification failed: {str(e)}'
            }


def run_migration(db_config):
    """
    Standalone function to run database migration
    
    Args:
        db_config (dict): Database configuration
        
    Returns:
        dict: Migration results
    """
    try:
        from . import get_database
        
        # Create database instance
        db = get_database(db_config)
        
        # Create migration instance
        migration = DatabaseMigration(db)
        
        # Run migration
        result = migration.migrate_to_3nf()
        
        if result['success']:
            # Verify migration
            verification = migration.verify_migration()
            result['verification'] = verification
            
            logger.info("Database migration completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running migration: {e}")
        return {
            'success': False,
            'message': f'Migration failed: {str(e)}'
        }


if __name__ == "__main__":
    # Example usage
    import sys
    import os
    
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Example SQLite migration
    sqlite_config = {
        'type': 'sqlite',
        'path': 'reviision.db'
    }
    
    result = run_migration(sqlite_config)
    print(f"Migration result: {result}") 