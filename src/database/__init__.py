"""
Database module for Retail Analytics System
Provides database interfaces for SQLite storage backend
"""

from .sqlite_db import SQLiteDatabase

def get_database(config):
    """
    Factory function to create database instance based on configuration
    
    Args:
        config (dict): Database configuration dictionary
        
    Returns:
        SQLiteDatabase: SQLite database instance
        
    Raises:
        ValueError: If database type is not sqlite or not specified
    """
    db_type = config.get('type', '').lower()
    
    if db_type == 'sqlite' or db_type == '':
        return SQLiteDatabase(config)
    else:
        raise ValueError(f"Unsupported database type: {db_type}. Only SQLite is supported.") 