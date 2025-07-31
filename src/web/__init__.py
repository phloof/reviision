"""
Web module for ReViision
Handles the Flask web application and API endpoints for retail vision analytics
"""

# Added SocketIO for real-time zone update broadcasts
from flask import Flask, jsonify, request, render_template, redirect, url_for
from flask_socketio import SocketIO, emit
from datetime import datetime, timedelta
import os
from .routes import web_bp  # Import the Blueprint from routes.py
from .auth import AuthService  # Import the authentication service
from utils.auth_setup import AuthSetup  # Import authentication setup utilities

def create_app(config, db):
    """
    Create and configure the Flask application
    
    Args:
        config (dict): Configuration dictionary
        db: Database instance
        
    Returns:
        Flask: Configured Flask application
    """
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))

    # ------------------------------------------------------------------
    # Real-time WebSocket support (zones updates & future events)
    # ------------------------------------------------------------------
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    app.socketio = socketio  # expose for routes
    
    # Configure app with explicit loop=True for video looping
    app.config['CAMERA_OPTIONS'] = {
        'loop': True,  # Ensure video looping is enabled by default
        'playback_speed': config.get('camera', {}).get('playback_speed', 1.0)
    }
    
    # Configure static file versioning for cache busting
    app.config['STATIC_VERSION'] = '5'  # Increment this to force cache reload
    
    # Configure session security
    app.config['SECRET_KEY'] = config.get('web', {}).get('secret_key', 'dev-key-change-in-production')
    app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS only
    app.config['SESSION_COOKIE_HTTPONLY'] = True  # No JS access
    app.config['SESSION_COOKIE_SAMESITE'] = 'Strict'  # CSRF protection
    
    # Store database reference for route handlers
    app.db = db
    
    # Set database reference on analysis service for direct access
    from .services import analysis_service
    analysis_service.set_database(db)
    
    # Initialize authentication service
    app.auth_service = AuthService(db)
    
    # Initialize authentication setup (creates default admin if configured)
    auth_setup = AuthSetup(db, config)
    auth_setup.initialize_authentication()
    
    @app.route('/api/status', methods=['GET'])
    def get_status():
        """Get system status"""
        status = {
            'status': 'running',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'cameras': {
                'active': 1,
                'total': 1
            },
            'detections': {
                'today': 120,
                'last_hour': 15
            }
        }
        return jsonify(status)
    
    @app.route('/api/system/config', methods=['GET'])
    def get_config():
        """Get system configuration (public fields only)"""
        public_config = {
            'web': config.get('web', {}),
            'analysis': {
                'heatmap': config.get('analysis', {}).get('heatmap', {})
            }
        }
        return jsonify(public_config)
    
    # Register the Blueprint from routes.py
    app.register_blueprint(web_bp, url_prefix='')

    # Helper to run SocketIO in main if needed (used by main.py)
    def run_with_socketio(*args, **kwargs):
        # Set allow_unsafe_werkzeug=True by default if not specified
        if 'allow_unsafe_werkzeug' not in kwargs:
            kwargs['allow_unsafe_werkzeug'] = True
        socketio.run(app, *args, **kwargs)
    app.run_with_socketio = run_with_socketio

    return app 