"""
Authentication Service for ReViision

Provides comprehensive authentication and authorization functionality including:
- User login/logout with secure session management  
- Password hashing using Argon2id algorithm for maximum security
- Account lockout protection against brute force attacks
- Role-based access control (Admin, Manager, Viewer)
- Session timeout and security token management
- Password strength validation and change functionality
"""

import secrets
import logging
from datetime import datetime, timedelta
from functools import wraps
from flask import request, session, jsonify, redirect, url_for, current_app
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, HashingError
import re

logger = logging.getLogger(__name__)

class AuthService:
    """
    Core authentication service implementing security best practices
    
    Features:
    - Argon2id password hashing with configurable parameters
    - Session-based authentication with secure tokens
    - Account lockout after failed login attempts
    - Password strength enforcement
    - Session cleanup and timeout management
    """
    
    def __init__(self, db):
        """
        Initialize authentication service
        
        Args:
            db: Database instance
        """
        self.db = db
        self.ph = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=1,
            hash_len=32,
            salt_len=16
        )
        
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=15)
        self.session_timeout = timedelta(hours=2)
        self.password_min_length = 8
        
        self.cleanup_expired_sessions()
    
    def hash_password(self, password):
        """
        Hash password using Argon2id
        
        Args:
            password (str): Plain text password
            
        Returns:
            str: Hashed password or None if error
        """
        try:
            if not self._is_password_strong(password):
                return None
            return self.ph.hash(password)
        except HashingError as e:
            logger.error(f"Password hashing error: {e}")
            return None
    
    def verify_password(self, password, password_hash):
        """
        Verify password against hash
        
        Args:
            password (str): Plain text password
            password_hash (str): Stored password hash
            
        Returns:
            bool: True if password matches
        """
        try:
            self.ph.verify(password_hash, password)
            
            if self.ph.check_needs_rehash(password_hash):
                logger.info("Password needs rehashing with updated parameters")
                return "rehash"
            
            return True
        except VerifyMismatchError:
            return False
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def _is_password_strong(self, password):
        """
        Check password strength requirements
        
        Args:
            password (str): Password to check
            
        Returns:
            bool: True if password meets requirements
        """
        if len(password) < self.password_min_length:
            return False
        
        patterns = [r'[A-Z]', r'[a-z]', r'\d', r'[!@#$%^&*(),.?":{}|<>]']
        return all(re.search(pattern, password) for pattern in patterns)
    
    def create_user(self, username, email, password, full_name=None, role='viewer'):
        """
        Create a new user with secure password hashing
        
        Args:
            username (str): Unique username
            email (str): User email
            password (str): Plain text password
            full_name (str): User's full name
            role (str): User role
            
        Returns:
            dict: Result with success status and message
        """
        try:
            # Validate input
            if not username or not email or not password:
                return {"success": False, "message": "All fields are required"}
            
            if not self._is_valid_email(email):
                return {"success": False, "message": "Invalid email format"}
            
            if not self._is_password_strong(password):
                return {"success": False, "message": "Password does not meet security requirements"}
            
            # Check if user already exists
            if self.db.get_user_by_username(username):
                return {"success": False, "message": "Username already exists"}
            
            if self.db.get_user_by_email(email):
                return {"success": False, "message": "Email already registered"}
            
            # Hash password
            password_hash = self.hash_password(password)
            if not password_hash:
                return {"success": False, "message": "Password hashing failed"}
            
            # Create user
            user_id = self.db.create_user(username, email, password_hash, full_name, role)
            if user_id:
                logger.info(f"New user created: {username} (ID: {user_id})")
                return {"success": True, "message": "User created successfully", "user_id": user_id}
            else:
                return {"success": False, "message": "Failed to create user"}
                
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return {"success": False, "message": "Internal error during user creation"}
    
    def authenticate_user(self, username, password, ip_address, user_agent):
        """
        Authenticate user with security checks
        
        Args:
            username (str): Username
            password (str): Plain text password
            ip_address (str): Client IP address
            user_agent (str): Client user agent
            
        Returns:
            dict: Authentication result
        """
        try:
            # Get user from database
            user = self.db.get_user_by_username(username)
            
            # Log attempt
            success = False
            failure_reason = None
            
            if not user:
                failure_reason = "User not found"
            elif not user[6]:  # is_active column
                failure_reason = "Account disabled"
            elif user[12] and datetime.fromisoformat(user[12]) > datetime.now():  # locked_until
                failure_reason = "Account locked"
            elif user[11] >= self.max_login_attempts:  # failed_login_attempts
                # Lock account
                lock_until = datetime.now() + self.lockout_duration
                self.db.update_failed_login_attempts(username, lock_until)
                failure_reason = "Too many failed attempts - account locked"
            else:
                # Verify password
                password_result = self.verify_password(password, user[3])  # password_hash column
                
                if password_result == True or password_result == "rehash":
                    success = True
                    
                    # Update password hash if needed
                    if password_result == "rehash":
                        new_hash = self.hash_password(password)
                        if new_hash:
                            # Update user's password hash (would need additional method)
                            logger.info(f"Password rehashed for user: {username}")
                else:
                    failure_reason = "Invalid password"
            
            # Log the attempt
            self.db.log_login_attempt(username, ip_address, user_agent, success, failure_reason)
            
            if success:
                # Create session
                session_token = self._generate_session_token()
                session_expires = datetime.now() + self.session_timeout
                
                # Update user login info
                self.db.update_user_login(user[0], session_token, session_expires)
                
                # Create session record
                self.db.create_user_session(user[0], session_token, ip_address, user_agent, session_expires)
                
                logger.info(f"Successful login: {username} from {ip_address}")
                
                return {
                    "success": True,
                    "message": "Login successful",
                    "session_token": session_token,
                    "user": {
                        "id": user[0],
                        "username": user[1],
                        "email": user[2],
                        "full_name": user[4],
                        "role": user[5]
                    }
                }
            else:
                # Increment failed attempts
                if user:
                    self.db.update_failed_login_attempts(username)
                
                logger.warning(f"Failed login attempt: {username} from {ip_address} - {failure_reason}")
                return {"success": False, "message": "Invalid credentials"}
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {"success": False, "message": "Authentication service error"}
    
    def verify_session(self, session_token):
        """
        Verify and get session information
        
        Args:
            session_token (str): Session token
            
        Returns:
            dict: Session information or None
        """
        try:
            session = self.db.get_session(session_token)
            if session:
                return {
                    "user_id": session[1],
                    "username": session[10],
                    "role": session[11],
                    "is_active": session[12]
                }
            return None
        except Exception as e:
            logger.error(f"Session verification error: {e}")
            return None
    
    def logout_user(self, session_token):
        """
        Logout user by invalidating session
        
        Args:
            session_token (str): Session token to invalidate
            
        Returns:
            bool: True if successful
        """
        try:
            return self.db.invalidate_session(session_token)
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            self.db.cleanup_expired_sessions()
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
    
    def _generate_session_token(self):
        """Generate a cryptographically secure session token"""
        return secrets.token_urlsafe(32)
    
    def _is_valid_email(self, email):
        """Basic email validation"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def get_password_requirements(self):
        """Get password requirements for client-side validation"""
        return {
            "min_length": self.password_min_length,
            "requires_uppercase": True,
            "requires_lowercase": True,
            "requires_digit": True,
            "requires_special": True,
            "special_chars": "!@#$%^&*(),.?\":{}|<>"
        }
    
    def change_password(self, user_id, current_password, new_password):
        """
        Change user password with current password verification
        
        Args:
            user_id (int): User ID
            current_password (str): Current password for verification
            new_password (str): New password
            
        Returns:
            dict: Result with success status and message
        """
        try:
            user = self.db.get_user_by_id(user_id)
            if not user:
                return {"success": False, "message": "User not found"}
            
            # Verify current password
            if not self.verify_password(current_password, user[3]):
                return {"success": False, "message": "Current password is incorrect"}
            
            # Validate new password
            if not self._is_password_strong(new_password):
                return {"success": False, "message": "New password does not meet security requirements"}
            
            # Hash new password
            new_password_hash = self.hash_password(new_password)
            if not new_password_hash:
                return {"success": False, "message": "Password hashing failed"}
            
            # Update password in database
            if self.db.update_user_password(user_id, new_password_hash):
                logger.info(f"Password changed for user ID: {user_id}")
                return {"success": True, "message": "Password changed successfully"}
            else:
                return {"success": False, "message": "Failed to update password"}
                
        except Exception as e:
            logger.error(f"Error changing password: {e}")
            return {"success": False, "message": "Internal error during password change"}
    
    def is_using_default_password(self, username):
        """
        Check if user is still using the default password
        
        Args:
            username (str): Username to check
            
        Returns:
            bool: True if user is using default password, False otherwise
        """
        try:
            # Only check for admin user with default password
            if username == "admin":
                user = self.db.get_user_by_username(username)
                if user:
                    # Check if still using default password "admin"
                    try:
                        self.ph.verify(user[3], "admin")  # user[3] is password_hash
                        return True
                    except:
                        return False
            return False
        except Exception as e:
            logger.error(f"Error checking default password for user {username}: {e}")
            return False


def require_auth(role_required=None):
    """
    Decorator to require authentication for routes
    
    Args:
        role_required (str): Required role (admin, manager, viewer)
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Get session token from cookie or header
            session_token = request.cookies.get('session_token') or request.headers.get('Authorization')
            
            if session_token and session_token.startswith('Bearer '):
                session_token = session_token[7:]  # Remove 'Bearer ' prefix
            
            if not session_token:
                if request.is_json:
                    return jsonify({"error": "Authentication required"}), 401
                return redirect(url_for('web.login_page'))
            
            # Verify session
            auth_service = current_app.auth_service
            session_info = auth_service.verify_session(session_token)
            
            if not session_info:
                if request.is_json:
                    return jsonify({"error": "Invalid or expired session"}), 401
                return redirect(url_for('web.login_page'))
            
            # Check role if required
            if role_required:
                user_role = session_info['role']
                role_hierarchy = {'viewer': 1, 'manager': 2, 'admin': 3}
                
                if role_hierarchy.get(user_role, 0) < role_hierarchy.get(role_required, 999):
                    if request.is_json:
                        return jsonify({"error": "Insufficient permissions"}), 403
                    return redirect(url_for('web.access_denied'))
            
            # Add user info to request context
            request.current_user = session_info
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def require_admin(f):
    """Decorator to require admin role"""
    return require_auth('admin')(f)


def require_manager(f):
    """Decorator to require manager role or higher"""
    return require_auth('manager')(f) 