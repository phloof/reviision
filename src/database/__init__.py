"""
Database module for Retail Analytics System
Provides database interfaces for different storage backends
"""

from .sqlite_db import SQLiteDatabase
from .mongodb import MongoDBDatabase

def get_database(config):
    """
    Factory function to create database instance based on configuration
    
    Args:
        config (dict): Database configuration dictionary
        
    Returns:
        Database: Database instance based on the specified type
        
    Raises:
        ValueError: If database type is not supported
    """
    db_type = config.get('type', '').lower()
    
    if db_type == 'sqlite':
        return SQLiteDatabase(config)
    elif db_type == 'mongodb':
        return MongoDBDatabase(config)
    else:
        raise ValueError(f"Unsupported database type: {db_type}") 