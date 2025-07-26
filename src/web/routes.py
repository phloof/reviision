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
from pathlib import Path
import time
import cv2

# Create logger early so it can be used in import exception handlers
logger = logging.getLogger(__name__)

# Import camera module
from camera import get_camera, stop_camera
from .services import analysis_service
from utils.config import ConfigManager

# Import analysis modules
from analysis.dwell import DwellTimeAnalyzer
from analysis.heatmap import HeatmapGenerator

# Import authentication
from .auth import require_auth, require_admin, require_manager

# Create a Blueprint for web routes
web_bp = Blueprint('web', __name__, template_folder='templates')

# Favicon route
@web_bp.route('/favicon.ico')
def favicon():
    """Serve favicon for browsers requesting /favicon.ico"""
    from flask import send_from_directory
    import os
    icon_path = os.path.join(os.path.dirname(__file__), 'static', 'img')
    return send_from_directory(icon_path, 'favicon.ico', mimetype='image/vnd.microsoft.icon')

def get_config_manager():
    """
    Get a properly configured ConfigManager instance
    
    Returns:
        ConfigManager: Configured instance pointing to the correct config.yaml
    """
    config_path = Path(__file__).parent.parent / 'config.yaml'
    return ConfigManager(config_dir=str(config_path.parent))

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

@web_bp.route('/api/analytics/summary', methods=['GET'])
@require_auth()
def get_analytics_summary():
    """
    Get analytics summary data with optional demographic filtering
    """
    try:
        hours = int(request.args.get('hours', 24))
        age_filter = request.args.get('age_filter', None)
        gender_filter = request.args.get('gender_filter', None)

        summary_data = analysis_service.get_analytics_summary(
            hours=hours,
            age_filter=age_filter,
            gender_filter=gender_filter
        )
        return jsonify({"success": True, **summary_data})
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@web_bp.route('/api/analytics/traffic', methods=['GET'])
@require_auth()
def get_traffic_data():
    """
    Get traffic data for charts with optional demographic filtering
    """
    try:
        hours = int(request.args.get('hours', 24))
        age_filter = request.args.get('age_filter', None)
        gender_filter = request.args.get('gender_filter', None)

        traffic_data = analysis_service.get_traffic_data(
            hours=hours,
            age_filter=age_filter,
            gender_filter=gender_filter
        )
        return jsonify({"success": True, **traffic_data})
    except Exception as e:
        logger.error(f"Error getting traffic data: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@web_bp.route('/api/analytics/demographics', methods=['GET'])
@require_auth()
def get_demographics_data():
    """
    Get detailed demographics data for table
    """
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        search = request.args.get('search', '')
        sort_by = request.args.get('sort_by', 'timestamp')
        sort_order = request.args.get('sort_order', 'desc')
        hours = int(request.args.get('hours', 24))

        demographics_data = analysis_service.get_demographics_data(
            page=page,
            per_page=per_page,
            search=search,
            sort_by=sort_by,
            sort_order=sort_order,
            hours=hours
        )
        return jsonify({"success": True, **demographics_data})
    except Exception as e:
        logger.error(f"Error getting demographics data: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@web_bp.route('/api/analytics/demographic-trends', methods=['GET'])
@require_auth()
def get_demographic_trends():
    """
    Get demographic trends over time for charts
    """
    try:
        hours = int(request.args.get('hours', 24))
        trends_data = analysis_service.get_demographic_trends(hours=hours)
        return jsonify(trends_data)
    except Exception as e:
        logger.error(f"Error getting demographic trends: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ============================================================================
# Export Routes (CSV / JSON)
# ============================================================================

import io
import csv


def _build_csv_from_mapping(mapping: dict) -> str:
    """Helper to convert a simple label:value dict to CSV string"""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["label", "value"])
    for k, v in mapping.items():
        writer.writerow([k, v])
    return output.getvalue()


@web_bp.route('/api/export/<chart_id>.json', methods=['GET'])
@require_auth()
def export_chart_json(chart_id):
    """Return JSON payload for a given chart_id (age_distribution, gender_distribution, demographic_trends)"""
    try:
        hours = int(request.args.get('hours', 24))
        summary = analysis_service.get_analytics_summary(hours=hours)

        if chart_id == 'age_distribution':
            payload = summary.get('age_groups', {})
        elif chart_id == 'gender_distribution':
            payload = summary.get('gender_distribution', {})
        elif chart_id == 'demographic_trends':
            payload = summary.get('traffic', {})
        else:
            return jsonify({"success": False, "message": "Unknown chart id"}), 400

        return jsonify({"success": True, "data": payload})
    except Exception as e:
        logger.error(f"Export JSON error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500


@web_bp.route('/api/export/<chart_id>.csv', methods=['GET'])
@require_auth()
def export_chart_csv(chart_id):
    """Return CSV file for a given chart_id"""
    try:
        hours = int(request.args.get('hours', 24))
        summary = analysis_service.get_analytics_summary(hours=hours)

        if chart_id == 'age_distribution':
            data_map = summary.get('age_groups', {})
        elif chart_id == 'gender_distribution':
            data_map = summary.get('gender_distribution', {})
        elif chart_id == 'demographic_trends':
            traffic = summary.get('traffic', {})
            # traffic may be in labels/data list format
            labels = traffic.get('labels', [])
            values = traffic.get('data', [])
            data_map = dict(zip(labels, values))
        else:
            return jsonify({"success": False, "message": "Unknown chart id"}), 400

        csv_str = _build_csv_from_mapping(data_map)
        buf = io.BytesIO(csv_str.encode('utf-8'))
        fname = f"{chart_id}.csv"
        return Response(buf.getvalue(), mimetype='text/csv', headers={
            'Content-Disposition': f'attachment;filename={fname}'
        })
    except Exception as e:
        logger.error(f"Export CSV error: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@web_bp.route('/api/analyze_frame', methods=['POST'])
def analyze_frame():
    """
    Analyze a video frame and return detection results with rate limiting
    This endpoint processes image data and returns detected objects, demographics, etc.
    """
    try:
        data = request.get_json()
        if not data or 'image_data' not in data:
            return jsonify({"error": "Missing image data"}), 400
        
        # Use the frame analysis service with built-in rate limiting
        result = analysis_service.analyze_frame(data['image_data'])
        
        # Handle rate limiting errors by returning 429 status
        if "error" in result:
            if "rate limited" in result["error"].lower() or "in progress" in result["error"].lower():
                return jsonify(result), 429  # Too Many Requests
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        return jsonify({"error": str(e)}), 500


@web_bp.route('/api/face_thumbnail/<int:person_id>', methods=['GET'])
def get_face_thumbnail(person_id):
    """Get face thumbnail for a person"""
    try:
        size = request.args.get('size', '64x64')
        width, height = map(int, size.split('x'))
        
        # Get face thumbnail from the analysis service
        if hasattr(analysis_service, 'face_snapshot_manager') and analysis_service.face_snapshot_manager:
            thumbnail_base64 = analysis_service.face_snapshot_manager.get_face_thumbnail_base64(
                person_id, (width, height)
            )
            
            if thumbnail_base64:
                return jsonify({
                    'success': True,
                    'thumbnail': thumbnail_base64,
                    'person_id': person_id,
                    'size': f"{width}x{height}"
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No face image found for person',
                    'person_id': person_id
                }), 404
        else:
            return jsonify({
                'success': False,
                'error': 'Face snapshot manager not available'
            }), 503
            
    except ValueError:
        return jsonify({
            'success': False,
            'error': 'Invalid size format. Use format: WIDTHxHEIGHT (e.g., 64x64)'
        }), 400
    except Exception as e:
        logger.error(f"Error getting face thumbnail for person {person_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@web_bp.route('/api/person/<int:person_id>/face_info', methods=['GET'])
def get_person_face_info(person_id):
    """Get face snapshot information for a person"""
    try:
        if not hasattr(analysis_service, 'face_snapshot_manager') or not analysis_service.face_snapshot_manager:
            return jsonify({
                'success': False,
                'error': 'Face snapshot manager not available'
            }), 503
        
        # Get face snapshot info from database
        db = current_app.db
        primary_face = db.get_primary_face_snapshot(person_id)
        
        if primary_face:
            face_info = {
                'person_id': person_id,
                'has_face': True,
                'quality_score': primary_face.get('quality_score', 0.0),
                'quality_grade': analysis_service.face_snapshot_manager.quality_scorer.get_quality_grade(
                    primary_face.get('quality_score', 0.0)
                ),
                'confidence': primary_face.get('confidence', 0.0),
                'timestamp': primary_face.get('timestamp'),
                'analysis_method': primary_face.get('analysis_method', 'unknown'),
                'face_size': f"{primary_face.get('face_width', 0)}x{primary_face.get('face_height', 0)}",
                'is_primary': primary_face.get('is_primary', False)
            }
        else:
            face_info = {
                'person_id': person_id,
                'has_face': False,
                'quality_score': 0.0,
                'quality_grade': 'N/A'
            }
        
        return jsonify({
            'success': True,
            'face_info': face_info
        })
        
    except Exception as e:
        logger.error(f"Error getting face info for person {person_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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

# Legacy mock endpoints - replaced with real analytics endpoints
# @web_bp.route('/api/facial_analysis', methods=['POST'])
# @web_bp.route('/api/traffic_patterns', methods=['GET'])
# These endpoints have been replaced with:
# - /api/analytics/summary
# - /api/analytics/traffic
# - /api/populate-sample-data

# ============================================================================
# Video and Camera Configuration Routes
# ============================================================================

@web_bp.route('/camera_stream')
def camera_stream():
    """Stream video from the configured camera (uses main config.yaml settings)"""
    try:
        # Load main configuration to get camera settings
        config_manager = get_config_manager()
        config_path = Path(__file__).parent.parent / 'config.yaml'
        config = config_manager.load_config(str(config_path))
        
        camera_config = config.get('camera', {})
        
        # Add credential manager to camera config for secure access
        camera_config['credential_manager'] = config_manager.get_credential_manager()
        
        logger.info(f"Starting camera stream with config: {camera_config.get('type', 'unknown')} camera")
        
        # Create camera instance using main configuration
        camera = get_camera(camera_config)
        
        if not camera.is_running:
            camera.start()
        
        def generate():
            """Generate video frames from the configured camera"""
            while True:
                try:
                    frame = camera.get_frame()
                    if frame is not None:
                        # Encode frame as JPEG
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if ret:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        else:
                            logger.warning("Failed to encode frame")
                    else:
                        logger.warning("No frame received from camera; retrying in 5s")
                        time.sleep(5.0)
                except Exception as e:
                    logger.error(f"Error in camera stream: {e}")
                    time.sleep(0.1)
                    break
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
    except Exception as e:
        logger.error(f"Error starting camera stream: {e}")
        return f"Camera stream error: {e}", 500

# ---------------------------------------------------------------------------
# Snapshot endpoint for zone editor
# ---------------------------------------------------------------------------

@web_bp.route('/api/camera/snapshot')
def camera_snapshot():
    """Return a single JPEG frame from the current camera feed"""
    try:
        from camera import _camera_manager
        
        # Check if camera manager exists
        if not hasattr(_camera_manager, '_current_camera') or not _camera_manager._current_camera:
            logger.warning("Camera snapshot: No active camera found")
            return "No active camera", 503
            
        cam = _camera_manager._current_camera
        
        # Check if camera is running
        if not cam.is_running:
            logger.warning("Camera snapshot: Camera not running, attempting to start")
            cam.start()
            time.sleep(0.5)  # Give it a moment to start
        
        frame = cam.get_frame()
        if frame is None:
            logger.warning("Camera snapshot: No frame available from camera")
            return "No frame available", 503
            
        ret, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            logger.error("Camera snapshot: Failed to encode frame as JPEG")
            return "Encode error", 500
            
        logger.info(f"Camera snapshot: Successfully captured frame ({len(buf)} bytes)")
        return Response(buf.tobytes(), mimetype='image/jpeg', headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        })
    except Exception as e:
        logger.error(f"Snapshot error: {e}", exc_info=True)
        return f"Snapshot error: {e}", 500

@web_bp.route('/api/camera/status')
def camera_status():
    """Get camera status for debugging"""
    try:
        from camera import _camera_manager
        
        status = {
            'camera_manager_exists': hasattr(_camera_manager, '_current_camera'),
            'current_camera_exists': False,
            'camera_running': False,
            'camera_type': None,
            'error': None
        }
        
        if hasattr(_camera_manager, '_current_camera') and _camera_manager._current_camera:
            cam = _camera_manager._current_camera
            status['current_camera_exists'] = True
            status['camera_running'] = cam.is_running
            status['camera_type'] = type(cam).__name__
            
            # Try to get a frame
            try:
                frame = cam.get_frame()
                status['frame_available'] = frame is not None
                if frame is not None:
                    status['frame_shape'] = frame.shape
            except Exception as e:
                status['frame_error'] = str(e)
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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

@web_bp.route('/api/camera/restart', methods=['POST'])
def restart_camera():
    """Restart camera with current configuration"""
    try:
        logger.info("Camera restart requested")
        
        # Stop current camera
        from camera import reset_camera
        reset_camera()
        logger.info("Camera fully reset")
        
        # Small delay to ensure cleanup
        time.sleep(2)
        
        # Load fresh configuration
        config_manager = get_config_manager()
        config_path = Path(__file__).parent.parent / 'config.yaml'
        
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return jsonify({
                'status': 'error', 
                'message': 'Configuration file not found'
            }), 500
        
        config = config_manager.load_config(str(config_path))
        camera_config = config.get('camera', {})
        
        if not camera_config:
            logger.error("No camera configuration found")
            return jsonify({
                'status': 'error', 
                'message': 'No camera configuration found'
            }), 500
        
        camera_config['credential_manager'] = config_manager.get_credential_manager()
        
        # Start camera with new configuration
        camera = get_camera(camera_config)
        if not camera.is_running:
            camera.start()
            
        # Verify camera is working
        time.sleep(1)
        test_frame = camera.get_frame()
        if test_frame is None:
            logger.warning("Camera restarted but no frame received")
        
        logger.info(f"Camera restarted successfully with {camera_config.get('type', 'unknown')} configuration")
        return jsonify({
            'status': 'success', 
            'message': f'Camera restarted with {camera_config.get("type", "unknown")} configuration'
        })
        
    except Exception as e:
        logger.error(f"Error restarting camera: {e}", exc_info=True)
        return jsonify({
            'status': 'error', 
            'message': f'Failed to restart camera: {str(e)}'
        }), 500

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
        # Use a basic config for colormap retrieval
        config = {
            'resolution': (640, 480),
            'alpha': 0.6,
            'blur_radius': 15,
            'point_decay': 0.99,
            'max_accumulate_frames': 300
        }
        heatmap_gen = HeatmapGenerator(config)
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
        # Load configuration using helper function
        config_manager = get_config_manager()
        config_path = Path(__file__).parent.parent / 'config.yaml'
        config = config_manager.load_config(str(config_path))
        
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
        
        # Verify config file exists
        if not config_path.exists():
            return jsonify({'status': 'error', 'message': f'Configuration file not found: {config_path}'}), 404
            
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
        config_manager = get_config_manager()
        config_path = Path(__file__).parent.parent / 'config.yaml'
        config = config_manager.load_config(str(config_path))
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
        config_manager = get_config_manager()
        config_path = Path(__file__).parent.parent / 'config.yaml'
        config = config_manager.load_config(str(config_path))
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
        config_manager = get_config_manager()
        config_path = Path(__file__).parent.parent / 'config.yaml'
        config = config_manager.load_config(str(config_path))
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
        import json
        from datetime import datetime
        
        config_manager = get_config_manager()
        config_path = Path(__file__).parent.parent / 'config.yaml'
        config = config_manager.load_config(str(config_path))
        
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

@web_bp.route('/api/data/populate-sample', methods=['POST'])
@require_admin
def populate_sample_data():
    """Populate database with sample data for testing/demo purposes"""
    try:
        from datetime import datetime, timedelta
        import random

        db = current_app.db
        
        # Generate sample data for the last 7 days
        sample_data = []
        current_time = datetime.now()

        # Sample demographics data
        age_groups = [
            {'min': 13, 'max': 17, 'group': 'Teen'},
            {'min': 18, 'max': 25, 'group': 'Young Adult'},
            {'min': 26, 'max': 35, 'group': 'Adult'},
            {'min': 36, 'max': 50, 'group': 'Middle Age'},
            {'min': 51, 'max': 65, 'group': 'Senior'},
            {'min': 66, 'max': 80, 'group': 'Elderly'}
        ]

        genders = ['Male', 'Female']
        emotions = ['Happy', 'Neutral', 'Sad', 'Angry', 'Surprised', 'Fear']
        races = ['Asian', 'Black', 'White', 'Hispanic', 'Indian', 'Middle Eastern']

        # Generate 100-200 sample records over 7 days
        total_records = random.randint(100, 200)

        for i in range(total_records):
            # Random timestamp within last 7 days
            days_ago = random.randint(0, 6)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)

            timestamp = current_time - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)

            # Random demographics
            age_group = random.choice(age_groups)
            age = random.randint(age_group['min'], age_group['max'])
            gender = random.choice(genders)
            emotion = random.choice(emotions)
            race = random.choice(races)

            # Random bounding box (simulated detection)
            x1 = random.randint(50, 300)
            y1 = random.randint(50, 200)
            x2 = x1 + random.randint(50, 150)
            y2 = y1 + random.randint(100, 200)

            confidence = random.uniform(0.7, 0.95)

            # Store in database
            try:
                # Insert detection record first
                detection_id = db.store_detection(
                    person_id=i,  # Use loop index as person_id
                    bbox=[x1, y1, x2, y2],
                    confidence=confidence,
                    timestamp=timestamp
                )
                
                # Insert demographics record
                demo_id = db.store_demographics(
                    person_id=i,
                    demographics={
                        'age': age,
                        'gender': gender,
                        'race': race,
                        'emotion': emotion,
                        'confidence': confidence
                    },
                    timestamp=timestamp,
                    detection_id=detection_id
                )
                
                person_id = i

                if person_id:
                    sample_data.append({
                        'person_id': person_id,
                        'timestamp': timestamp.isoformat(),
                        'age': age,
                        'gender': gender,
                        'race': race,
                        'emotion': emotion
                    })

            except Exception as e:
                logger.warning(f"Failed to insert sample record {i}: {e}")
                continue

        logger.info(f"Successfully populated {len(sample_data)} sample records")

        return jsonify({
            'success': True,
            'message': f'Successfully added {len(sample_data)} sample records',
            'records_created': len(sample_data)
        })

    except Exception as e:
        logger.error(f"Error populating sample data: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to populate sample data: {str(e)}'
        }), 500

@web_bp.route('/api/data/clear-demographics', methods=['POST'])
@require_admin
def clear_demographics_data():
    """Clear all demographic data from database"""
    try:
        db = current_app.db

        # Get count before clearing
        conn = db._get_connection()
        cursor = conn.cursor()

        # Count existing records
        cursor.execute("SELECT COUNT(*) FROM demographics")
        demo_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM persons")
        person_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM detections")
        detection_count = cursor.fetchone()[0]

        # Clear demographic data
        cursor.execute("DELETE FROM demographics")
        cursor.execute("DELETE FROM dwell_times")
        cursor.execute("DELETE FROM detections")
        cursor.execute("DELETE FROM persons")

        # Reset auto-increment counters
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('demographics', 'persons', 'detections', 'dwell_times')")

        conn.commit()

        logger.info(f"Cleared {demo_count} demographic records, {person_count} person records, {detection_count} detection records")

        return jsonify({
            'success': True,
            'message': 'Successfully cleared all customer data',
            'records_cleared': {
                'demographics': demo_count,
                'persons': person_count,
                'detections': detection_count
            }
        })
        
    except Exception as e:
        logger.error(f"Error clearing demographics data: {e}")
        return jsonify({
            'success': False,
            'message': f'Failed to clear demographics data: {str(e)}'
        }), 500

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

@web_bp.route('/api/insert_initial_data', methods=['POST'])
@require_auth('admin')
def insert_initial_data():
    """Insert initial data: default users and sample data"""
    try:
        db = current_app.db
        db._create_default_users()
        sample_result = db.populate_sample_data()
        return jsonify({
            'status': 'success',
            'message': 'Initial data inserted successfully',
            'details': sample_result
        })
    except Exception as e:
        logger.error(f"Error inserting initial data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@web_bp.route('/api/clear_all_data', methods=['POST'])
@require_auth('admin')
def clear_all_data():
    """Clear all analytics and user data (except admin)"""
    try:
        db = current_app.db
        # Delete all data from main tables except admin user
        conn = db._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM dwell_times")
        cursor.execute("DELETE FROM demographics")
        cursor.execute("DELETE FROM detections")
        cursor.execute("DELETE FROM persons")
        cursor.execute("DELETE FROM users WHERE username != 'admin'")
        conn.commit()
        db._create_default_users()  # Re-create default users (except admin, which is preserved)
        return jsonify({'status': 'success', 'message': 'All data cleared. Default users restored (admin, manager, viewer, manager2, viewer2).'})
    except Exception as e:
        logger.error(f"Error clearing all data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============================================================================
# User Settings Compatibility Endpoints
# ============================================================================

@web_bp.route('/api/user_info', methods=['GET'])
@require_auth()
def get_user_info():
    """Get current user information (compatibility endpoint for user settings page)"""
    try:
        user_info = request.current_user
        return jsonify({
            "success": True,
            "username": user_info['username'],
            "role": user_info['role']
        })
    except Exception as e:
        logger.error(f"Get user info error: {e}")
        return jsonify({"success": False, "message": "User service error"}), 500

@web_bp.route('/api/check_default_password', methods=['GET'])
@require_auth()
def check_default_password():
    """Check if user is using default password (compatibility endpoint)"""
    try:
        user_info = request.current_user
        auth_service = current_app.auth_service
        is_default = auth_service.is_using_default_password(user_info['username'])
        
        return jsonify({
            "success": True,
            "is_default": is_default
        })
    except Exception as e:
        logger.error(f"Check default password error: {e}")
        return jsonify({"success": False, "message": "Service error"}), 500

@web_bp.route('/change_password', methods=['POST'])
@require_auth()
def change_password():
    """Change user password (compatibility endpoint for user settings page)"""
    try:
        # Handle both JSON and form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        if not data:
            return jsonify({"success": False, "message": "No data provided"}), 400
        
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        if not current_password or not new_password:
            return jsonify({"success": False, "message": "Both current and new passwords required"}), 400
        
        user_info = request.current_user
        auth_service = current_app.auth_service
        result = auth_service.change_password(user_info['user_id'], current_password, new_password)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Change password error: {e}")
        return jsonify({"success": False, "message": "Password change service error"}), 500

# ============================================================================
# PTZ Control Routes
# ============================================================================

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

@web_bp.route('/api/analytics/historical', methods=['GET'])
@require_auth()
def get_historical_analytics():
    """
    Get combined historical analytics data for the historical page
    This endpoint combines summary and traffic data into a single response
    """
    try:
        hours = int(request.args.get('hours', 24))

        # Get analytics summary
        summary_data = analysis_service.get_analytics_summary(hours=hours)

        # Get traffic data
        traffic_data = analysis_service.get_traffic_data(hours=hours)

        # Get additional historical data
        weekly_patterns = analysis_service.database.get_weekly_patterns(hours=hours)
        peak_hour_analysis = analysis_service.database.get_peak_hour_analysis_by_day(days=30)
        historical_demographics = analysis_service.database.get_historical_demographics(hours=hours)
        dwell_time_stats = analysis_service.database.get_historical_dwell_time_stats(hours=hours)

        # Combine the data as expected by the frontend
        combined_data = {
            "success": True,
            "total_visitors": summary_data.get("total_visitors", 0),
            "avg_dwell_time": summary_data.get("avg_dwell_time", 0),
            "conversion_rate": summary_data.get("conversion_rate", 0),
            "peak_hour": summary_data.get("peak_hour", "--:--"),
            "gender_distribution": summary_data.get("gender_distribution", {}),
            "age_groups": summary_data.get("age_groups", {}),
            "emotions": summary_data.get("emotions", {}),
            "races": summary_data.get("races", {}),
            "avg_age": summary_data.get("avg_age", 0),
            "traffic": traffic_data,
            "weekly_traffic": traffic_data,  # Frontend expects this field
            "weekly_patterns": weekly_patterns,
            "peak_hour_analysis": peak_hour_analysis,
            "historical_demographics": historical_demographics,
            "dwell_time_stats": dwell_time_stats,
            "dwell_time_distribution": dwell_time_stats.get("distribution", {
                "labels": ["0-2 min", "2-5 min", "5-10 min", "10-20 min", "20+ min"],
                "data": [0, 0, 0, 0, 0]
            }),
            "start_time": summary_data.get("start_time", ""),
            "end_time": summary_data.get("end_time", "")
        }

        return jsonify(combined_data)
    except Exception as e:
        logger.error(f"Error getting historical analytics: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@web_bp.route('/api/analytics/weekly-patterns', methods=['GET'])
@require_auth()
def get_weekly_patterns():
    """
    Get weekly visitor patterns data
    """
    try:
        hours = int(request.args.get('hours', 24))

        # Get weekly patterns data from database
        weekly_data = analysis_service.database.get_weekly_patterns(hours=hours)

        return jsonify(weekly_data)
    except Exception as e:
        logger.error(f"Error getting weekly patterns: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@web_bp.route('/api/analytics/peak-hour-analysis', methods=['GET'])
@require_auth()
def get_peak_hour_analysis():
    """
    Get peak hour analysis by day of week
    """
    try:
        days = int(request.args.get('days', 30))

        # Get peak hour analysis data from database
        peak_hour_data = analysis_service.database.get_peak_hour_analysis_by_day(days=days)

        return jsonify(peak_hour_data)
    except Exception as e:
        logger.error(f"Error getting peak hour analysis: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@web_bp.route('/api/analytics/historical-demographics', methods=['GET'])
@require_auth()
def get_historical_demographics():
    """
    Get historical demographics data
    """
    try:
        hours = int(request.args.get('hours', 24))

        # Get historical demographics data from database
        demographics_data = analysis_service.database.get_historical_demographics(hours=hours)

        return jsonify(demographics_data)
    except Exception as e:
        logger.error(f"Error getting historical demographics: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

@web_bp.route('/api/analytics/historical-dwell-time', methods=['GET'])
@require_auth()
def get_historical_dwell_time():
    """
    Get historical dwell time statistics
    """
    try:
        hours = int(request.args.get('hours', 24))

        # Get historical dwell time data from database
        dwell_time_data = analysis_service.database.get_historical_dwell_time_stats(hours=hours)

        return jsonify(dwell_time_data)
    except Exception as e:
        logger.error(f"Error getting historical dwell time: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# ============================================================================
# Zone Management Routes
# ============================================================================

@web_bp.route('/zones')
@require_auth('manager')
def zones_editor():
    """Render the zone editor page"""
    return render_template('zones.html')


@web_bp.route('/api/zones', methods=['GET'])
@require_auth()
def api_get_zones():
    """Return list of zones for current camera (single camera setup)"""
    zones = current_app.zone_manager.get_zones()
    return jsonify({"success": True, "zones": zones})


@web_bp.route('/api/zones', methods=['POST'])
@require_auth('manager')
def api_create_zone():
    data = request.get_json()
    required = {'name', 'x1', 'y1', 'x2', 'y2'}
    if not data or not required.issubset(data):
        return jsonify({"success": False, "message": "Missing fields"}), 400
    zone_id = current_app.zone_manager.create_zone(
        data['name'], data['x1'], data['y1'], data['x2'], data['y2']
    )
    # Broadcast update
    if hasattr(current_app, 'socketio'):
        current_app.socketio.emit('zonesUpdated')
    return jsonify({"success": True, "zone_id": zone_id})


@web_bp.route('/api/zones/<int:zone_id>', methods=['PUT'])
@require_auth('manager')
def api_update_zone(zone_id):
    data = request.get_json() or {}
    if not data:
        return jsonify({"success": False, "message": "No data"}), 400
    current_app.zone_manager.update_zone(zone_id, **data)
    if hasattr(current_app, 'socketio'):
        current_app.socketio.emit('zonesUpdated')
    return jsonify({"success": True})


@web_bp.route('/api/zones/<int:zone_id>', methods=['DELETE'])
@require_auth('manager')
def api_delete_zone(zone_id):
    current_app.zone_manager.delete_zone(zone_id)
    if hasattr(current_app, 'socketio'):
        current_app.socketio.emit('zonesUpdated')
    return jsonify({"success": True})


# ============================================================================
# PDF Report Generation Routes
# ============================================================================

@web_bp.route('/api/reports/demographics/pdf', methods=['GET'])
@require_auth()
def generate_demographics_pdf_report():
    """Generate comprehensive demographics PDF report"""
    try:
        # Get parameters
        hours = int(request.args.get('hours', 24))
        logger.info(f"Generating demographics PDF report for {hours} hours")

        # Get data from analysis service - use the same pattern as historical report
        summary_data = analysis_service.get_analytics_summary(hours=hours)

        # Also get demographic trends data for more comprehensive report
        try:
            demographic_trends = analysis_service.get_demographic_trends(hours=hours)
        except Exception as e:
            logger.warning(f"Could not get demographic trends: {e}")
            demographic_trends = {}

        logger.info(f"Summary data success: {summary_data.get('success', 'unknown')}")
        logger.info(f"Summary data keys: {list(summary_data.keys()) if isinstance(summary_data, dict) else 'not dict'}")

        if not summary_data.get('success', True):
            logger.error(f"Failed to retrieve analytics data: {summary_data.get('message', 'Unknown error')}")
            return jsonify({
                'success': False,
                'message': 'Failed to retrieve analytics data'
            }), 500

        # Combine data similar to historical report pattern
        combined_data = {
            **summary_data,
            'demographic_trends': demographic_trends,
            'period_hours': hours
        }

        logger.info(f"Combined data keys: {list(combined_data.keys())}")

        # Import report generator
        from analysis.reports import DemographicsReportGenerator

        # Generate time period description
        if hours <= 24:
            time_period = f"Last {hours} hours"
        elif hours <= 168:  # 7 days
            days = hours // 24
            time_period = f"Last {days} day{'s' if days > 1 else ''}"
        else:
            days = hours // 24
            time_period = f"Last {days} days"

        logger.info(f"Generating report for time period: {time_period}")

        # Generate report with detailed logging
        logger.info("Creating DemographicsReportGenerator instance...")
        report_generator = DemographicsReportGenerator()
        logger.info("DemographicsReportGenerator created successfully")

        logger.info("Starting PDF generation...")
        logger.info(f"Data summary: total_visitors={combined_data.get('total_visitors', 0)}, "
                   f"age_groups={len(combined_data.get('age_groups', {}))}, "
                   f"gender_distribution={len(combined_data.get('gender_distribution', {}))}")

        try:
            pdf_bytes = report_generator.generate_report(combined_data, time_period)
            logger.info("PDF generation completed")
        except Exception as e:
            logger.error(f"Error during PDF generation: {e}", exc_info=True)
            raise Exception(f"PDF generation failed: {str(e)}")

        if not pdf_bytes:
            logger.error("Report generator returned empty PDF")
            raise Exception("Report generator returned empty PDF")

        logger.info(f"PDF generated successfully, size: {len(pdf_bytes)} bytes")

        # Create response
        response = Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="demographics_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"',
                'Content-Type': 'application/pdf'
            }
        )

        logger.info(f"Demographics PDF report generated successfully for {hours} hours period")
        return response

    except Exception as e:
        logger.error(f"Error generating demographics PDF report: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f'Report generation failed: {str(e)}'
        }), 500


@web_bp.route('/api/reports/historical/pdf', methods=['GET'])
@require_auth()
def generate_historical_pdf_report():
    """Generate comprehensive historical analytics PDF report"""
    try:
        # Get parameters
        hours = int(request.args.get('hours', 24))

        # Get data from analysis service (using historical endpoint data structure)
        summary_data = analysis_service.get_analytics_summary(hours=hours)
        traffic_data = analysis_service.get_traffic_data(hours=hours)

        if not summary_data.get('success', True) or not traffic_data.get('success', True):
            return jsonify({
                'success': False,
                'message': 'Failed to retrieve analytics data'
            }), 500

        # Combine data as expected by historical report
        combined_data = {
            **summary_data,
            'traffic': traffic_data,
            'period_hours': hours
        }

        # Import report generator
        from analysis.reports import HistoricalReportGenerator

        # Generate time period description
        if hours <= 24:
            time_period = f"Last {hours} hours"
        elif hours <= 168:  # 7 days
            days = hours // 24
            time_period = f"Last {days} day{'s' if days > 1 else ''}"
        else:
            days = hours // 24
            time_period = f"Last {days} days"

        # Generate report
        report_generator = HistoricalReportGenerator()
        pdf_bytes = report_generator.generate_report(combined_data, time_period)

        # Create response
        response = Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename="historical_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf"',
                'Content-Type': 'application/pdf'
            }
        )

        logger.info(f"Historical PDF report generated successfully for {hours} hours period")
        return response

    except Exception as e:
        logger.error(f"Error generating historical PDF report: {e}")
        return jsonify({
            'success': False,
            'message': f'Report generation failed: {str(e)}'
        }), 500
