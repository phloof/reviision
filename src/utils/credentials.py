"""
Credential Manager for ReViision
Securely stores and retrieves sensitive credentials
"""

import os
import json
import base64
import logging
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

class CredentialManager:
    """
    Secure credential manager that handles encryption and decryption of sensitive data
    
    This class provides methods to securely store and retrieve credentials such as
    RTSP usernames, passwords, API keys, and other sensitive configuration.
    """
    
    def __init__(self, config_dir='./config', key_env_var='REVIISION_KEY'):
        """
        Initialize the credential manager
        
        Args:
            config_dir (str): Directory to store encrypted credential files
            key_env_var (str): Environment variable name for the encryption key
        """
        self.config_dir = Path(config_dir)
        self.key_env_var = key_env_var
        self.credentials_file = self.config_dir / 'credentials.enc'
        self.salt_file = self.config_dir / '.salt'
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption key
        self.key = self._get_or_create_key()
        self.fernet = Fernet(self.key) if self.key else None
        
        # Cache for credentials
        self._credential_cache = {}
        
        if not self.fernet:
            logger.warning(f"Credentials will not be encrypted. Set {key_env_var} environment variable for encryption.")
    
    def _get_or_create_key(self):
        """
        Get or create encryption key from environment variable or generate a new one
        
        Returns:
            bytes: Encryption key or None if not available and not generating
        """
        # Try to get key from environment variable
        env_key = os.environ.get(self.key_env_var)
        if env_key:
            try:
                # Try to decode the key
                return base64.urlsafe_b64decode(env_key)
            except Exception as e:
                logger.error(f"Invalid encryption key in environment variable: {e}")
        
        # If no environment key, try to generate using salt
        try:
            # Check if salt exists, create if not
            if not self.salt_file.exists():
                salt = os.urandom(16)
                with open(self.salt_file, 'wb') as f:
                    f.write(salt)
                # Secure the salt file
                os.chmod(self.salt_file, 0o600)
            else:
                with open(self.salt_file, 'rb') as f:
                    salt = f.read()
            
            # If we have env passphrase, derive key
            passphrase = os.environ.get('REVIISION_PASSPHRASE')
            if passphrase:
                # Derive key from passphrase and salt
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))
                return key
            
            # If still no key, generate random one for this session
            # Note: This is NOT secure for production, as key will change between runs
            logger.warning("Generating temporary encryption key. For persistent encryption, set REVIISION_KEY or REVIISION_PASSPHRASE environment variable.")
            return Fernet.generate_key()
            
        except Exception as e:
            logger.error(f"Error generating encryption key: {e}")
            return None
    
    def get_credential(self, service, key):
        """
        Get a credential from the encrypted store
        
        Args:
            service (str): Service identifier (e.g., 'rtsp', 'api')
            key (str): Credential key (e.g., 'username', 'password')
            
        Returns:
            str: Credential value or None if not found
        """
        # Use cached credentials if available
        if self._credential_cache and service in self._credential_cache and key in self._credential_cache[service]:
            return self._credential_cache[service][key]
        
        # Load credentials from file
        credentials = self._load_credentials()
        
        # Get credential if it exists
        if service in credentials and key in credentials[service]:
            return credentials[service][key]
        
        return None
    
    def set_credential(self, service, key, value):
        """
        Set a credential in the encrypted store
        
        Args:
            service (str): Service identifier (e.g., 'rtsp', 'api')
            key (str): Credential key (e.g., 'username', 'password')
            value (str): Credential value
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.fernet:
            logger.error("Cannot set credential: encryption key not available")
            return False
        
        try:
            # Load existing credentials
            credentials = self._load_credentials()
            
            # Initialize service dict if not exists
            if service not in credentials:
                credentials[service] = {}
            
            # Set credential
            credentials[service][key] = value
            
            # Update cache
            if service not in self._credential_cache:
                self._credential_cache[service] = {}
            self._credential_cache[service][key] = value
            
            # Save credentials
            return self._save_credentials(credentials)
            
        except Exception as e:
            logger.error(f"Error setting credential: {e}")
            return False
    
    def remove_credential(self, service, key=None):
        """
        Remove a credential or entire service from the encrypted store
        
        Args:
            service (str): Service identifier
            key (str, optional): Credential key to remove. If None, removes entire service
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load existing credentials
            credentials = self._load_credentials()
            
            # Check if service exists
            if service not in credentials:
                return False
            
            # Remove specific key or entire service
            if key is not None:
                if key in credentials[service]:
                    del credentials[service][key]
                    # Update cache
                    if service in self._credential_cache and key in self._credential_cache[service]:
                        del self._credential_cache[service][key]
                else:
                    return False
            else:
                del credentials[service]
                # Update cache
                if service in self._credential_cache:
                    del self._credential_cache[service]
            
            # Save credentials
            return self._save_credentials(credentials)
            
        except Exception as e:
            logger.error(f"Error removing credential: {e}")
            return False
    
    def _load_credentials(self):
        """
        Load and decrypt credentials from file
        
        Returns:
            dict: Credentials dictionary
        """
        if not self.credentials_file.exists():
            return {}
        
        try:
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
            
            if self.fernet:
                # Decrypt data
                decrypted_data = self.fernet.decrypt(encrypted_data)
                return json.loads(decrypted_data.decode('utf-8'))
            else:
                logger.warning("Reading credentials as plain text (no encryption key available)")
                try:
                    # Try to read as plain JSON if no encryption
                    return json.loads(encrypted_data.decode('utf-8'))
                except:
                    logger.error("Could not decode credentials file")
                    return {}
                
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return {}
    
    def _save_credentials(self, credentials):
        """
        Encrypt and save credentials to file
        
        Args:
            credentials (dict): Credentials dictionary
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert to JSON
            credentials_json = json.dumps(credentials)
            
            if self.fernet:
                # Encrypt data
                encrypted_data = self.fernet.encrypt(credentials_json.encode('utf-8'))
            else:
                # Store as plain text with warning
                logger.warning("Storing credentials as plain text (no encryption key available)")
                encrypted_data = credentials_json.encode('utf-8')
            
            # Write to file
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Secure the file
            os.chmod(self.credentials_file, 0o600)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            return False
    
    def import_from_env(self, prefix='RA_CRED_'):
        """
        Import credentials from environment variables
        
        Environment variables should be in the format:
        RA_CRED_SERVICE_KEY=value (e.g., RA_CRED_RTSP_USERNAME=admin)
        
        Args:
            prefix (str): Prefix for environment variables
            
        Returns:
            int: Number of credentials imported
        """
        count = 0
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Extract service and credential key
                parts = key[len(prefix):].split('_', 1)
                if len(parts) == 2:
                    service = parts[0].lower()
                    cred_key = parts[1].lower()
                    if self.set_credential(service, cred_key, value):
                        count += 1
        
        return count
    
    def get_service_credentials(self, service):
        """
        Get all credentials for a service
        
        Args:
            service (str): Service identifier
            
        Returns:
            dict: Service credentials or empty dict if not found
        """
        credentials = self._load_credentials()
        return credentials.get(service, {})
    
    def list_services(self):
        """
        List all services with stored credentials
        
        Returns:
            list: List of service identifiers
        """
        credentials = self._load_credentials()
        return list(credentials.keys()) 