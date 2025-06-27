"""
Authentication Setup Utilities

This module handles initial authentication setup tasks such as:
- Creating default admin users when specified in configuration
- Setting up initial authentication requirements
- Managing authentication initialization
"""

import logging
from argon2 import PasswordHasher

logger = logging.getLogger(__name__)


class AuthSetup:
    """Handles authentication setup and initialization tasks"""
    
    def __init__(self, database, config):
        """
        Initialize authentication setup
        
        Args:
            database: Database instance for user management
            config: Application configuration
        """
        self.database = database
        self.config = config
        self.ph = PasswordHasher()
    
    def initialize_authentication(self):
        """
        Initialize authentication system including default users if configured
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Check if default admin creation is enabled in config
            create_default = self.config.get('web', {}).get('create_default_admin', False)
            
            if create_default:
                self._create_default_admin_if_needed()
            
            logger.info("Authentication system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing authentication system: {e}")
            return False
    
    def _create_default_admin_if_needed(self):
        """
        Create default admin user if no users exist in the system
        
        This creates a user with:
        - Username: admin
        - Password: admin  
        - Role: admin
        
        Only creates the user if the database is empty of users.
        """
        try:
            # Check if any users exist
            user_count = self._get_user_count()
            
            if user_count == 0:
                # Create default admin user
                self._create_default_admin()
                logger.warning("Default admin user created - Username: admin, Password: admin")
                logger.warning("SECURITY WARNING: Please change the default password immediately!")
            else:
                logger.info(f"Users already exist ({user_count} found), skipping default admin creation")
                
        except Exception as e:
            logger.error(f"Error checking/creating default admin user: {e}")
    
    def _get_user_count(self):
        """
        Get total number of users in the system
        
        Returns:
            int: Number of users in database
        """
        try:
            # Use database method to count users
            if hasattr(self.database, 'cursor'):
                self.database.cursor.execute('SELECT COUNT(*) FROM users')
                return self.database.cursor.fetchone()[0]
            else:
                # Handle other database types if needed
                return 0
        except Exception as e:
            logger.error(f"Error counting users: {e}")
            return 0
    
    def _create_default_admin(self):
        """
        Create the default admin user in the database
        
        Raises:
            Exception: If user creation fails
        """
        try:
            # Hash the default password
            password_hash = self.ph.hash("admin")
            
            # Insert default admin user
            if hasattr(self.database, 'cursor'):
                self.database.cursor.execute('''
                    INSERT INTO users (username, email, password_hash, full_name, role)
                    VALUES (?, ?, ?, ?, ?)
                ''', ("admin", "admin@reviision.local", password_hash, "Default Administrator", "admin"))
                self.database.conn.commit()
            
            logger.info("Default admin user created successfully")
            
        except Exception as e:
            logger.error(f"Error creating default admin user: {e}")
            raise
    
    def change_default_password_warning(self, username):
        """
        Check if user should be warned about default password
        
        Args:
            username (str): Username to check
            
        Returns:
            bool: True if user should be warned about default password
        """
        try:
            if username == "admin":
                user = self.database.get_user_by_username(username)
                if user:
                    # Check if still using default password
                    try:
                        self.ph.verify(user[3], "admin")  # user[3] is password_hash
                        return True
                    except:
                        return False
            return False
        except Exception as e:
            logger.error(f"Error checking default password for user {username}: {e}")
            return False 