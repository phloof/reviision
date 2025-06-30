"""
Flask Routes for ReViision

Provides web interface and API endpoints for retail analytics including:
- Authentication endpoints (login, logout, registration)
- Main dashboard and analytics views
- Camera configuration and video streaming
- Settings and configuration management
"""

from flask import Blueprint, render_template, request, jsonify, current_app, Response, redirect, url_for
import logging
from datetime import datetime
import json
import os
from pathlib import Path
import time
import cv2
import numpy as np

# Create logger early so it can be used in import exception handlers
logger = logging.getLogger(__name__)

# Import camera module
from camera import get_camera, stop_camera
from .services import analysis_service
from utils.config import ConfigManager

# Import the path analyzer if available
try:
    from src.analysis.path import PathAnalyzer
except ImportError:
    logger.warning("PathAnalyzer not available, using mock interface")
    class PathAnalyzer:
        """Mock path analyzer for testing"""
        def __init__(self, config):
            pass
        
        def generate_mock_data(self):
            return {"message": "Path analyzer not available"}

# Import the dwell time analyzer if available
try:
    from src.analysis.dwell import DwellTimeAnalyzer
except ImportError:
    logger.warning("DwellTimeAnalyzer not available, using mock interface")
    class DwellTimeAnalyzer:
        """Mock dwell time analyzer for testing"""
        def __init__(self, config):
            pass
        
        def generate_mock_data(self):
            return {"message": "Dwell time analyzer not available"}

# Import authentication
from .auth import require_auth, require_admin, require_manager

# Create a Blueprint for web routes
web_bp = Blueprint('web', __name__, template_folder='templates')

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

@web_bp.route('/logout')
def logout():
    """Handle user logout and redirect to login page"""
    try:
        session_token = request.cookies.get('session_token')
        if session_token:
            auth_service = current_app.auth_service
            auth_service.logout_user(session_token)
        
        # Create response and clear the session cookie
        response = redirect(url_for('web.login_page'))
        response.set_cookie('session_token', '', expires=0)
        return response
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        # Even if there's an error, redirect to login and clear cookie
        response = redirect(url_for('web.login_page'))
        response.set_cookie('session_token', '', expires=0)
        return response

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

# ============================================================================
# ONVIF PTZ Control Routes
# ============================================================================

@web_bp.route('/api/config/camera', methods=['POST'])
@require_auth('manager')
def update_camera_config():
    """Update camera configuration"""
    try:
        data = request.get_json()
        if not data or 'camera' not in data:
            return jsonify({"success": False, "message": "No camera configuration provided"}), 400
        
        # Update config using existing method
        config_data = {'camera': data['camera']}
        response = update_main_config()
        return response
        
    except Exception as e:
        logger.error(f"Camera config update error: {e}")
        return jsonify({"success": False, "message": f"Failed to update camera configuration: {str(e)}"}), 500

@web_bp.route('/api/ptz/status', methods=['GET'])
@require_auth()
def get_ptz_status():
    """Get PTZ camera status and capabilities"""
    try:
        from camera import _camera_manager
        
        current_camera = _camera_manager._current_camera
        if not current_camera:
            return jsonify({"success": False, "message": "No camera active"}), 404
        
        # Check if it's an ONVIF camera with PTZ capabilities
        if hasattr(current_camera, 'get_ptz_status'):
            ptz_status = current_camera.get_ptz_status()
            camera_info = current_camera.get_camera_info()
            
            return jsonify({
                "success": True,
                "capabilities": ptz_status.get('capabilities', {}),
                "position": ptz_status.get('position', {}),
                "limits": ptz_status.get('limits', {}),
                "presets": ptz_status.get('presets', []),
                "camera_info": camera_info
            })
        else:
            return jsonify({
                "success": False,
                "message": "Current camera does not support PTZ",
                "capabilities": {"pan": False, "tilt": False, "zoom": False}
            })
    
    except Exception as e:
        logger.error(f"PTZ status error: {e}")
        return jsonify({"success": False, "message": f"Failed to get PTZ status: {str(e)}"}), 500

@web_bp.route('/api/ptz/absolute', methods=['POST'])
@require_auth()
def ptz_move_absolute():
    """Move PTZ camera to absolute position"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No position data provided"}), 400
        
        pan = float(data.get('pan', 0))
        tilt = float(data.get('tilt', 0))
        zoom = float(data.get('zoom', 0))
        
        from camera import _camera_manager
        
        current_camera = _camera_manager._current_camera
        if not current_camera or not hasattr(current_camera, 'move_absolute'):
            return jsonify({"success": False, "message": "PTZ not available"}), 404
        
        success = current_camera.move_absolute(pan, tilt, zoom)
        
        if success:
            return jsonify({"success": True, "message": "Camera moved successfully"})
        else:
            return jsonify({"success": False, "message": "Failed to move camera"}), 500
    
    except Exception as e:
        logger.error(f"PTZ absolute move error: {e}")
        return jsonify({"success": False, "message": f"Failed to move camera: {str(e)}"}), 500

@web_bp.route('/api/ptz/relative', methods=['POST'])
@require_auth()
def ptz_move_relative():
    """Move PTZ camera by relative amounts"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No movement data provided"}), 400
        
        pan_delta = float(data.get('pan_delta', 0))
        tilt_delta = float(data.get('tilt_delta', 0))
        zoom_delta = float(data.get('zoom_delta', 0))
        
        from camera import _camera_manager
        
        current_camera = _camera_manager._current_camera
        if not current_camera or not hasattr(current_camera, 'move_relative'):
            return jsonify({"success": False, "message": "PTZ not available"}), 404
        
        success = current_camera.move_relative(pan_delta, tilt_delta, zoom_delta)
        
        if success:
            return jsonify({"success": True, "message": "Camera moved successfully"})
        else:
            return jsonify({"success": False, "message": "Failed to move camera"}), 500
    
    except Exception as e:
        logger.error(f"PTZ relative move error: {e}")
        return jsonify({"success": False, "message": f"Failed to move camera: {str(e)}"}), 500

@web_bp.route('/api/ptz/continuous', methods=['POST'])
@require_auth()
def ptz_move_continuous():
    """Start continuous PTZ movement"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No movement data provided"}), 400
        
        pan_speed = float(data.get('pan_speed', 0))
        tilt_speed = float(data.get('tilt_speed', 0))
        zoom_speed = float(data.get('zoom_speed', 0))
        
        from camera import _camera_manager
        
        current_camera = _camera_manager._current_camera
        if not current_camera or not hasattr(current_camera, 'start_continuous_move'):
            return jsonify({"success": False, "message": "PTZ not available"}), 404
        
        success = current_camera.start_continuous_move(pan_speed, tilt_speed, zoom_speed)
        
        if success:
            return jsonify({"success": True, "message": "Continuous movement started"})
        else:
            return jsonify({"success": False, "message": "Failed to start movement"}), 500
    
    except Exception as e:
        logger.error(f"PTZ continuous move error: {e}")
        return jsonify({"success": False, "message": f"Failed to start continuous movement: {str(e)}"}), 500

@web_bp.route('/api/ptz/stop', methods=['POST'])
@require_auth()
def ptz_stop():
    """Stop PTZ movement"""
    try:
        from camera import _camera_manager
        
        current_camera = _camera_manager._current_camera
        if not current_camera or not hasattr(current_camera, 'stop_ptz'):
            return jsonify({"success": False, "message": "PTZ not available"}), 404
        
        success = current_camera.stop_ptz()
        
        if success:
            return jsonify({"success": True, "message": "PTZ movement stopped"})
        else:
            return jsonify({"success": False, "message": "Failed to stop PTZ"}), 500
    
    except Exception as e:
        logger.error(f"PTZ stop error: {e}")
        return jsonify({"success": False, "message": f"Failed to stop PTZ: {str(e)}"}), 500

@web_bp.route('/api/ptz/preset/goto', methods=['POST'])
@require_auth()
def ptz_goto_preset():
    """Move to a preset position"""
    try:
        data = request.get_json()
        if not data or 'preset_token' not in data:
            return jsonify({"success": False, "message": "No preset token provided"}), 400
        
        preset_token = data['preset_token']
        
        from camera import _camera_manager
        
        current_camera = _camera_manager._current_camera
        if not current_camera or not hasattr(current_camera, 'goto_preset'):
            return jsonify({"success": False, "message": "PTZ presets not available"}), 404
        
        success = current_camera.goto_preset(preset_token)
        
        if success:
            return jsonify({"success": True, "message": "Moved to preset successfully"})
        else:
            return jsonify({"success": False, "message": "Failed to move to preset"}), 500
    
    except Exception as e:
        logger.error(f"PTZ goto preset error: {e}")
        return jsonify({"success": False, "message": f"Failed to move to preset: {str(e)}"}), 500

@web_bp.route('/api/ptz/preset/save', methods=['POST'])
@require_auth()
def ptz_save_preset():
    """Save current position as a preset"""
    try:
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"success": False, "message": "No preset name provided"}), 400
        
        preset_name = data['name']
        
        from camera import _camera_manager
        
        current_camera = _camera_manager._current_camera
        if not current_camera or not hasattr(current_camera, 'set_preset'):
            return jsonify({"success": False, "message": "PTZ presets not available"}), 404
        
        preset_token = current_camera.set_preset(preset_name)
        
        if preset_token:
            return jsonify({
                "success": True,
                "message": "Preset saved successfully",
                "preset_token": preset_token
            })
        else:
            return jsonify({"success": False, "message": "Failed to save preset"}), 500
    
    except Exception as e:
        logger.error(f"PTZ save preset error: {e}")
        return jsonify({"success": False, "message": f"Failed to save preset: {str(e)}"}), 500

@web_bp.route('/api/ptz/preset/delete', methods=['POST'])
@require_auth()
def ptz_delete_preset():
    """Delete a preset"""
    try:
        data = request.get_json()
        if not data or 'preset_token' not in data:
            return jsonify({"success": False, "message": "No preset token provided"}), 400
        
        preset_token = data['preset_token']
        
        from camera import _camera_manager
        
        current_camera = _camera_manager._current_camera
        if not current_camera or not hasattr(current_camera, 'remove_preset'):
            return jsonify({"success": False, "message": "PTZ presets not available"}), 404
        
        success = current_camera.remove_preset(preset_token)
        
        if success:
            return jsonify({"success": True, "message": "Preset deleted successfully"})
        else:
            return jsonify({"success": False, "message": "Failed to delete preset"}), 500
    
    except Exception as e:
        logger.error(f"PTZ delete preset error: {e}")
        return jsonify({"success": False, "message": f"Failed to delete preset: {str(e)}"}), 500 