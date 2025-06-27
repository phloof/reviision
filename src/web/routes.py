"""
Flask Routes for ReViision

Provides web interface and API endpoints for retail analytics including:
- Authentication endpoints (login, logout, registration)
- Main dashboard and analytics views
- Camera configuration and video streaming
- Settings and configuration management
"""

from flask import Blueprint, render_template, request, jsonify, current_app, Response
import logging
from datetime import datetime
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np

# Import camera module
from camera import get_camera, stop_camera
from .services import analysis_service
from utils.config import ConfigManager

# Import the heatmap generator if available
try:
    from src.analysis.heatmap import HeatmapGenerator
except ImportError:
    # Simple interface placeholder
    class HeatmapGenerator:
        """Basic heatmap generator interface"""
        def __init__(self, config=None):
            self.config = config or {}
        
        def generate(self, data):
            """Generate a heatmap from real data"""
            return None
            
        def get_available_colormaps(self):
            """Get list of supported colormaps"""
            return [
                {"name": "jet", "category": "sequential"},
                {"name": "viridis", "category": "perceptual"},
                {"name": "plasma", "category": "perceptual"},
                {"name": "hot", "category": "sequential"}
            ]

# Import authentication
from .auth import require_auth, require_admin, require_manager

# Create a Blueprint for web routes
web_bp = Blueprint('web', __name__, template_folder='templates')
logger = logging.getLogger(__name__)

# ============================================================================
# Authentication Routes
# ============================================================================

@web_bp.route('/login')
def login_page():
    """Render the login page"""
    return render_template('login.html')

@web_bp.route('/register')
def register_page():
    """Render the registration page"""
    return render_template('register.html')

@web_bp.route('/api/auth/login', methods=['POST'])
def api_login():
    """Handle user login"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
        
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if not username or not password:
            return jsonify({"success": False, "message": "Username and password required"}), 400
        
        ip_address = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        user_agent = request.headers.get('User-Agent', '')
        auth_service = current_app.auth_service
        result = auth_service.authenticate_user(username, password, ip_address, user_agent)
        
        if result["success"]:
            response = jsonify(result)
            response.set_cookie(
                'session_token', 
                result['session_token'],
                max_age=7200,
                secure=request.is_secure,
                httponly=True,
                samesite='Strict'
            )
            return response
        else:
            return jsonify(result), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({"success": False, "message": "Login service error"}), 500

@web_bp.route('/api/auth/register', methods=['POST'])
def api_register():
    """Handle user registration"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
        
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        
        auth_service = current_app.auth_service
        result = auth_service.create_user(username, email, password, full_name)
        
        if result["success"]:
            return jsonify(result), 201
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({"success": False, "message": "Registration service error"}), 500

@web_bp.route('/api/auth/logout', methods=['POST'])
@require_auth()
def api_logout():
    """Handle user logout"""
    try:
        session_token = request.cookies.get('session_token')
        if session_token:
            auth_service = current_app.auth_service
            auth_service.logout_user(session_token)
        
        response = jsonify({"success": True, "message": "Logged out successfully"})
        response.set_cookie('session_token', '', expires=0)
        return response
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({"success": False, "message": "Logout service error"}), 500

@web_bp.route('/api/auth/me', methods=['GET'])
@require_auth()
def api_get_user():
    """Get current user information"""
    try:
        user_info = request.current_user
        auth_service = current_app.auth_service
        is_default_password = auth_service.is_using_default_password(user_info['username'])
        
        return jsonify({
            "success": True,
            "user": {
                "username": user_info['username'],
                "role": user_info['role'],
                "is_default_password": is_default_password
            }
        })
    except Exception as e:
        logger.error(f"Get user error: {e}")
        return jsonify({"success": False, "message": "User service error"}), 500

@web_bp.route('/api/auth/password-requirements', methods=['GET'])
def api_password_requirements():
    """Get password requirements for client-side validation"""
    try:
        auth_service = current_app.auth_service
        requirements = auth_service.get_password_requirements()
        return jsonify({"success": True, "requirements": requirements})
    except Exception as e:
        logger.error(f"Password requirements error: {e}")
        return jsonify({"success": False, "message": "Service error"}), 500

@web_bp.route('/access-denied')
def access_denied():
    """Render access denied page"""
    return render_template('access_denied.html'), 403

@web_bp.route('/api/auth/change-password', methods=['POST'])
@require_auth()
def api_change_password():
    """Change user password"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
        
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        if not current_password or not new_password:
            return jsonify({"success": False, "message": "Both current and new passwords required"}), 400
        
        user_info = request.current_user
        auth_service = current_app.auth_service
        result = auth_service.change_password(user_info['user_id'], current_password, new_password)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Change password error: {e}")
        return jsonify({"success": False, "message": "Password change service error"}), 500

@web_bp.route('/user-settings')
@require_auth()
def user_settings():
    """Render the user settings page"""
    return render_template('user_settings.html')

# ============================================================================
# Main Page Routes
# ============================================================================

@web_bp.route('/')
@require_auth()
def index():
    """Render the dashboard homepage"""
    return render_template('index.html')

@web_bp.route('/heatmap')
@require_auth()
def heatmap():
    """Render the store heatmap visualization page"""
    return render_template('heatmap.html')

@web_bp.route('/analysis')
@require_auth()
def analysis():
    """Render the real-time analysis page with toggleable overlays"""
    return render_template('analysis.html')

@web_bp.route('/demographics')
@require_auth()
def demographics():
    """Render the demographics page"""
    return render_template('demographics.html')

@web_bp.route('/historical')
@require_auth()
def historical():
    """Render the historical data page"""
    return render_template('historical.html')

@web_bp.route('/settings')
@require_auth('manager')
def settings():
    """Render the settings page - requires manager role"""
    return render_template('settings.html')

# ============================================================================
# API Routes - Analysis
# ============================================================================

@web_bp.route('/api/analyze_frame', methods=['POST'])
def analyze_frame():
    """
    Analyze a video frame and return detection results
    This endpoint processes image data and returns detected objects, demographics, etc.
    """
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"error": "Missing image data"}), 400
        
        # Use the frame analysis service
        result = analysis_service.analyze_frame(data['image_data'])
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        return jsonify({"error": str(e)}), 500

@web_bp.route('/api/heatmap', methods=['GET'])
def api_heatmap():
    """Generate heatmap data for the specified time range"""
    try:
        start_time_str = request.args.get('start_time', '')
        end_time_str = request.args.get('end_time', '')
        
        # Parse time if provided, otherwise use defaults
        try:
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        try:
            end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            end_time = datetime.now()
        
        # Generate mock heatmap data for now
        grid_size = 20
        heatmap_data = []
        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                # Generate some realistic-looking heatmap values
                value = max(0, random.gauss(50, 30))
                row.append(min(100, value))
            heatmap_data.append(row)
        
        return jsonify({
            'status': 'success',
            'data': heatmap_data,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'grid_size': grid_size
        })
    
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@web_bp.route('/api/camera_feed', methods=['GET'])
def camera_feed():
    """Get camera feed data"""
    try:
        camera_id = request.args.get('camera_id', 'main')
        
        return jsonify({
            'status': 'success',
            'camera_id': camera_id,
            'message': 'Camera feed API endpoint ready'
        })
    except Exception as e:
        logger.error(f"Error fetching camera feed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@web_bp.route('/api/facial_analysis', methods=['POST'])
def facial_analysis():
    """Process image and return facial analysis data"""
    try:
        data = request.get_json()
        
        # Mock facial analysis data
        return jsonify({
            'status': 'success',
            'demographics': {
                'age_range': '25-35',
                'gender': 'unknown',
                'emotion': 'neutral'
            },
            'confidence': 0.75
        })
    except Exception as e:
        logger.error(f"Error in facial analysis: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@web_bp.route('/api/traffic_patterns', methods=['GET'])
def get_traffic_patterns():
    """Get traffic pattern analysis data"""
    try:
        # Mock traffic pattern data
        patterns = {
            'hourly_traffic': [10, 15, 25, 45, 60, 80, 90, 85, 70, 55, 40, 30],
            'common_paths': [
                {'path': 'entrance->electronics->checkout', 'frequency': 45},
                {'path': 'entrance->clothing->fitting_room->checkout', 'frequency': 30},
                {'path': 'entrance->grocery->checkout', 'frequency': 25}
            ],
            'avg_dwell_time': {'electronics': 180, 'clothing': 240, 'grocery': 120}
        }
        
        return jsonify({
            'status': 'success',
            'data': patterns
        })
    except Exception as e:
        logger.error(f"Error getting traffic patterns: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# Video and Camera Configuration Routes
# ============================================================================

@web_bp.route('/get_video_config')
def get_video_config():
    """Return the video configuration as JSON"""
    try:
        config_path = Path(__file__).parent / 'static' / 'video_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@web_bp.route('/video_stream/<video_id>')
def video_stream(video_id):
    """Stream video from the camera module"""
    try:
        # Load video configuration
        config_path = Path(__file__).parent / 'static' / 'video_config.json'
        
        with open(config_path, 'r') as f:
            video_config = json.load(f)
        
        # Get video path from config
        if video_id not in video_config['videos']:
            logger.warning(f"Video ID {video_id} not found, using default: {video_config['default']}")
            video_id = video_config['default']
        
        video_path = video_config['videos'][video_id]
        file_path = Path(video_path)
        
        # Try different path resolutions
        if not file_path.exists():
            web_root_path = Path(__file__).parent
            relative_path = Path(video_path.lstrip('/').lstrip('\\')).as_posix()
            file_path = web_root_path / relative_path
            
            if not file_path.exists():
                project_root = web_root_path.parent.parent
                file_path = project_root / relative_path
        
        if not file_path.exists():
            logger.error(f"Video file not found: {file_path}")
            return f"Video file not found: {file_path}", 404
        
        # Get camera options
        camera_options = current_app.config.get('CAMERA_OPTIONS', {})
        
        # Configure camera with video file
        camera_config = {
            'type': 'video_file',
            'file_path': str(file_path),
            'loop': camera_options.get('loop', True),
            'playback_speed': camera_options.get('playback_speed', 1.0)
        }
        
        # Create camera instance
        camera = get_camera(camera_config)
        
        if not camera.is_running:
            camera.start()
        
        def generate_frames():
            frame_count = 0
            max_frames = 10000  # Limit frames to prevent infinite streaming
            
            try:
                while frame_count < max_frames:
                    if not camera.is_running:
                        logger.info("Camera stopped, ending frame generation")
                        break
                        
                    frame = camera.get_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                    
                    _, buffer = cv2.imencode('.jpg', frame)
                    jpeg_frame = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame + b'\r\n')
                    
                    frame_count += 1
                    time.sleep(0.03)  # ~30fps
            
            except GeneratorExit:
                logger.info("Client disconnected from video stream")
            except Exception as e:
                logger.error(f"Error in frame generator: {e}")
            finally:
                # Don't stop the camera here - let the camera manager handle it
                logger.info(f"Frame generator ended after {frame_count} frames")
        
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    except Exception as e:
        logger.error(f"Error streaming video: {e}")
        return f"Error streaming video: {e}", 500

@web_bp.route('/update_video_config', methods=['POST'])
def update_video_config():
    """Update video configuration"""
    try:
        data = request.get_json()
        config_path = Path(__file__).parent / 'static' / 'video_config.json'
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update config with new data
        if 'default' in data:
            config['default'] = data['default']
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return jsonify({"success": True, "message": "Video configuration updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@web_bp.route('/update_camera_options', methods=['POST'])
def update_camera_options():
    """Update camera options in app config"""
    try:
        data = request.get_json()
        current_app.config['CAMERA_OPTIONS'] = data
        return jsonify({"success": True, "message": "Camera options updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@web_bp.route('/get_camera_options')
def get_camera_options():
    """Get current camera options"""
    try:
        options = current_app.config.get('CAMERA_OPTIONS', {
            'loop': True,
            'playback_speed': 1.0
        })
        return jsonify(options)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@web_bp.route('/stop_camera', methods=['POST'])
def stop_camera_route():
    """Stop the current camera instance"""
    try:
        stop_camera()
        return jsonify({"success": True, "message": "Camera stopped successfully"})
    except Exception as e:
        logger.error(f"Error stopping camera: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# Colormap Configuration Routes
# ============================================================================

@web_bp.route('/get_colormap_config')
def get_colormap_config():
    """Get colormap configuration"""
    try:
        config_path = Path(__file__).parent / 'static' / 'colormap_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@web_bp.route('/update_colormap_config', methods=['POST'])
def update_colormap_config():
    """Update colormap configuration"""
    try:
        data = request.get_json()
        config_path = Path(__file__).parent / 'static' / 'colormap_config.json'
        
        # Load existing config
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        
        # Update with new data
        config.update(data)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return jsonify({"success": True, "message": "Colormap configuration updated"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@web_bp.route('/api/available_colormaps')
def available_colormaps():
    """Get list of available colormaps"""
    try:
        # Initialize heatmap generator to get colormaps
        heatmap_gen = HeatmapGenerator()
        colormaps = heatmap_gen.get_available_colormaps()
        
        return jsonify({
            "success": True,
            "colormaps": colormaps
        })
    except Exception as e:
        logger.error(f"Error getting available colormaps: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "colormaps": [
                {"name": "jet", "category": "sequential"},
                {"name": "viridis", "category": "perceptual"},
                {"name": "plasma", "category": "perceptual"},
                {"name": "hot", "category": "sequential"}
            ]
        })

@web_bp.route('/api/set_heatmap_colormap', methods=['POST'])
def set_heatmap_colormap():
    """Set the heatmap colormap"""
    try:
        data = request.get_json()
        colormap_name = data.get('colormap', 'jet')
        
        # Update colormap config
        config_path = Path(__file__).parent / 'static' / 'colormap_config.json'
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        
        config['current_colormap'] = colormap_name
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return jsonify({
            "success": True,
            "message": f"Heatmap colormap set to {colormap_name}"
        })
    except Exception as e:
        logger.error(f"Error setting heatmap colormap: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ============================================================================
# Main Configuration Management Routes
# ============================================================================

@web_bp.route('/api/config', methods=['GET'])
def get_main_config():
    """Get main configuration from config.yaml"""
    try:
        from utils.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Return safe config (without sensitive data)
        safe_config = config_manager.get_safe_config()
        return jsonify({
            'status': 'success',
            'config': safe_config
        })
    except Exception as e:
        logger.error(f"Error loading main config: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@web_bp.route('/api/config', methods=['POST'])
def update_main_config():
    """Update main configuration file"""
    try:
        from utils.config import ConfigManager
        import yaml
        from pathlib import Path
        
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        # Load current config
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'r') as f:
            current_config = yaml.safe_load(f)
        
        # Update specific sections
        if 'detection' in data:
            current_config['detection'].update(data['detection'])
        if 'tracking' in data:
            current_config['tracking'].update(data['tracking'])
        if 'analysis' in data:
            for key, value in data['analysis'].items():
                if key in current_config['analysis']:
                    current_config['analysis'][key].update(value)
                else:
                    current_config['analysis'][key] = value
        if 'web' in data:
            current_config['web'].update(data['web'])
        if 'database' in data:
            current_config['database'].update(data['database'])
        if 'camera' in data:
            current_config['camera'].update(data['camera'])
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(current_config, f, default_flow_style=False, indent=2)
        
        return jsonify({'status': 'success', 'message': 'Configuration updated successfully'})
        
    except Exception as e:
        logger.error(f"Error updating main config: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@web_bp.route('/api/config/detection', methods=['GET'])
def get_detection_config():
    """Get detection configuration"""
    try:
        from utils.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config()
        return jsonify({
            'status': 'success',
            'config': config.get('detection', {})
        })
    except Exception as e:
        logger.error(f"Error loading detection config: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@web_bp.route('/api/config/tracking', methods=['GET'])
def get_tracking_config():
    """Get tracking configuration"""
    try:
        from utils.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config()
        return jsonify({
            'status': 'success',
            'config': config.get('tracking', {})
        })
    except Exception as e:
        logger.error(f"Error loading tracking config: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@web_bp.route('/api/config/analysis', methods=['GET'])
def get_analysis_config():
    """Get analysis configuration"""
    try:
        from utils.config import ConfigManager
        config_manager = ConfigManager()
        config = config_manager.load_config()
        return jsonify({
            'status': 'success',
            'config': config.get('analysis', {})
        })
    except Exception as e:
        logger.error(f"Error loading analysis config: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@web_bp.route('/api/config/backup', methods=['GET'])
def backup_config():
    """Create a backup of the current configuration"""
    try:
        from utils.config import ConfigManager
        import json
        from datetime import datetime
        
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Create backup with timestamp
        backup_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'config': config
        }
        
        return jsonify({
            'status': 'success',
            'backup': backup_data,
            'filename': f"reviision_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        })
    except Exception as e:
        logger.error(f"Error creating config backup: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@web_bp.route('/api/config/restore', methods=['POST'])
def restore_config():
    """Restore configuration from backup"""
    try:
        import yaml
        from pathlib import Path
        
        data = request.get_json()
        if not data or 'config' not in data:
            return jsonify({'status': 'error', 'message': 'Invalid backup data'}), 400
        
        # Save restored config
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(data['config'], f, default_flow_style=False, indent=2)
        
        return jsonify({'status': 'success', 'message': 'Configuration restored successfully'})
        
    except Exception as e:
        logger.error(f"Error restoring config: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500 