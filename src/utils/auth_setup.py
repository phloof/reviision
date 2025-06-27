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
        logger.info("AuthSetup initialized with database type: %s", type(database).__name__)
    
    def initialize_authentication(self):
        """
        Initialize authentication system including default users if configured
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Starting authentication system initialization...")
            
            # Check if default admin creation is enabled in config
            create_default = self.config.get('web', {}).get('create_default_admin', False)
            logger.info("Create default admin setting: %s", create_default)
            
            if create_default:
                logger.info("Attempting to create default admin user...")
                self._create_default_admin_if_needed()
            else:
                logger.info("Default admin creation is disabled in configuration")
            
            logger.info("Authentication system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing authentication system: {e}", exc_info=True)
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
            logger.info("Checking if default admin user needs to be created...")
            
            # Check if any users exist
            user_count = self._get_user_count()
            logger.info("Current user count in database: %d", user_count)
            
            if user_count == 0:
                # Create default admin user
                logger.info("No users found, creating default admin user...")
                self._create_default_admin()
                logger.warning("Default admin user created - Username: admin, Password: admin")
                logger.warning("SECURITY WARNING: Please change the default password immediately!")
            else:
                logger.info(f"Users already exist ({user_count} found), skipping default admin creation")
                
        except Exception as e:
            logger.error(f"Error checking/creating default admin user: {e}", exc_info=True)
    
    def _get_user_count(self):
        """
        Get total number of users in the system
        
        Returns:
            int: Number of users in database
        """
        try:
            logger.debug("Attempting to count users in database...")
            
            # Use database method to count users
            if hasattr(self.database, 'cursor') and self.database.cursor:
                logger.debug("Using direct cursor access to count users")
                self.database.cursor.execute('SELECT COUNT(*) FROM users')
                result = self.database.cursor.fetchone()
                count = result[0] if result else 0
                logger.debug("User count query result: %s", count)
                return count
            elif hasattr(self.database, 'get_user_count'):
                logger.debug("Using database method to count users")
                return self.database.get_user_count()
            else:
                logger.warning("No method available to count users, assuming 0")
                return 0
                
        except Exception as e:
            logger.error(f"Error counting users: {e}", exc_info=True)
            return 0
    
    def _create_default_admin(self):
        """
        Create the default admin user in the database
        
        Raises:
            Exception: If user creation fails
        """
        try:
            logger.info("Creating default admin user...")
            
            # Hash the default password
            password_hash = self.ph.hash("admin")
            logger.debug("Password hashed successfully")
            
            # Try different methods to create the user
            if hasattr(self.database, 'create_user'):
                logger.debug("Using database create_user method")
                user_id = self.database.create_user(
                    username="admin",
                    email="admin@reviision.local", 
                    password_hash=password_hash,
                    full_name="Default Administrator",
                    role="admin"
                )
                if user_id:
                    logger.info("Default admin user created successfully with ID: %s", user_id)
                else:
                    raise Exception("create_user method returned None")
                    
            elif hasattr(self.database, 'cursor') and self.database.cursor:
                logger.debug("Using direct cursor to create user")
                self.database.cursor.execute('''
                    INSERT INTO users (username, email, password_hash, full_name, role)
                    VALUES (?, ?, ?, ?, ?)
                ''', ("admin", "admin@reviision.local", password_hash, "Default Administrator", "admin"))
                
                if hasattr(self.database, 'conn') and self.database.conn:
                    self.database.conn.commit()
                    logger.info("Default admin user created successfully via direct cursor")
                else:
                    raise Exception("No database connection available for commit")
            else:
                raise Exception("No method available to create user in database")
            
        except Exception as e:
            logger.error(f"Error creating default admin user: {e}", exc_info=True)
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
                if hasattr(self.database, 'get_user_by_username'):
                    user = self.database.get_user_by_username(username)
                elif hasattr(self.database, 'cursor') and self.database.cursor:
                    self.database.cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
                    user = self.database.cursor.fetchone()
                else:
                    logger.warning("No method available to get user by username")
                    return False
                    
                if user:
                    # Check if still using default password
                    try:
                        # user[3] should be password_hash column
                        password_hash = user[3] if len(user) > 3 else None
                        if password_hash:
                            self.ph.verify(password_hash, "admin")
                            return True
                    except Exception:
                        return False
            return False
        except Exception as e:
            logger.error(f"Error checking default password for user {username}: {e}")
            return False 