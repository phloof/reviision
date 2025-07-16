# Retail Analytics System Documentations

## Overview

This document provides a comprehensive overview of the retail analytics system. It details the system's purpose, architecture, technology stack, key design decisions, and considerations for security and data privacy. This system is designed to analyze customer behavior and store dynamics using data captured from cameras.

## Purpose

The primary purpose of this retail analytics system is to provide insights into customer behavior within a retail environment. This is achieved by:

*   Tracking customer movement patterns.
*   Analyzing dwell times in specific areas.
*   Understanding customer demographics.
*   Identifying correlations between different aspects of customer behavior.
*   Generating heatmaps to visualize customer density.

## Architecture

The system is designed with a modular architecture, comprising several key components:

*   **Camera Interfaces:** Handles the connection and data streaming from various types of cameras.
*   **Detection:** Employs Machine learning models for detecting objects of interest such as people.
*   **Tracker:** Tracks each object.
*   **Analysis:** Analyzes the detected objects and their behavior.
*   **Configuration:** Manages the configuration of the system.
*   **Utilities:** Support other modules with data and helper functions.

## Technology Stack

The system utilizes a range of technologies to achieve its goals:

*   **Programming Languages:**
    *   Python: The primary language for the system, chosen for its extensive libraries in data science, machine learning, and computer vision.
*   **Libraries and Frameworks:**
    *   Machine Learning: Libraries for object detection and tracking.
    *   Data Processing: Libraries for handling and processing data from cameras and the ML models.
    *   Computer Vision: Libraries for image and video processing.
    *   Potentially: OpenCV: Might be used for real-time video processing and analysis.
* **Configuration:**
    * json: Used for configuration of the system.

## Component Details

### Camera Interfaces (`src/camera/`)

*   **Purpose:** To provide a standardized interface for connecting to different types of cameras.
*   **Components:**
    *   `base.py`: Defines the base class for all camera interfaces.
    *   `rtsp_camera.py`: Handles cameras streaming via the RTSP protocol.
    *   `rtsps_camera.py`: Handles secure cameras streaming via the RTSPS protocol.
    *   `usb_camera.py`: Handles USB connected cameras.
*   **Functionality:**
    *   Establish connections to cameras.
    *   Receive and stream video feeds.
    *   Manage camera settings and configurations.

### Detection (`src/detection/`)

*   **Purpose:** To detect and track objects in the video feeds.
*   **Components:**
    *   `detector.py`: Contains classes and methods for object detection.
    *   `tracker.py`: Contains classes and methods for tracking each detected object.
*   **Functionality:**
    *   Employ object detection models to identify objects.
    *   Track objects across video frames.
    *   Store object movement data.

### Analysis (`src/analysis/`)

*   **Purpose:** To perform in-depth analysis of customer behavior.
*   **Components:**
    *   `correlation.py`: Analyzes correlations between different aspects of customer behavior.
    *   `demographics.py`: Extracts and analyzes demographic information.
    *   `dwell.py`: Calculates and analyzes dwell times in specific areas.
    *   `heatmap.py`: Generates heatmaps to visualize customer density.
    *   `path.py`: Analyzes customer movement patterns.
*   **Functionality:**
    *   Process data from the `Detection` module.
    *   Calculate dwell times and track movement.
    *   Analyze demographic information.
    *   Generate heatmaps for visualization.
    *   Analyze correlations in customer behaviors.

### Web Interface (`src/web/`)

*   **Purpose:** Provides the Flask-based web interface and API endpoints.
*   **Components:**
    *   `__init__.py`: Flask app factory with minimal core routes
    *   `routes.py`: Clean, organized route definitions (436 lines, down from 1338)
    *   `services.py`: Business logic layer for frame analysis and detection
    *   `templates/`: HTML templates for the web interface
    *   `static/`: Static assets (CSS, JS, configuration files)
*   **Functionality:**
    *   Serve web dashboard and visualization pages
    *   Provide REST API endpoints for real-time analysis
    *   Handle video streaming and camera configuration
    *   Process frame analysis requests through service layer

### Utilities (`src/utils/`)

*   **Purpose:** To provide utility functions and classes that support other components.
*   **Components:**
    *   `config.py`: Centralized configuration management with environment awareness
    *   `credentials.py`: Secure credential management with encryption
*   **Functionality:**
    *   Load and validate system configurations
    *   Manage encrypted credentials for cameras and external services
    *   Provide cross-platform path resolution

### Main Application (`src/main.py`)

*   **Purpose:** Entry point.
*   **Functionality:**
    * Orchestrates the entire system.

### Configuration (`config/config_schema.json`)

* **Purpose:** Manages the system configurations.
* **Functionality:**
    * The schema that validates the configuration.

### Installation (`INSTALL.md`)

*   **Purpose:** Guides the installation and setup.
*   **Functionality:**
    *   Instructions for setting up the environment.
    *   Dependency management.

### Requirements (`requirements.txt`)

*   **Purpose:** Lists all necessary libraries.
*   **Functionality:**
    *   Ensures consistent dependencies.

### Readme (`README.md`)

* **Purpose**: General information about the system.
* **Functionality**:
    * Provides an overview of the project.
    * Explains the project's main purpose.
    * Includes any instructions or details.

## Key Design Decisions

*   **Modularity:** The system is designed with a modular approach, allowing for independent development and maintenance of components.
*   **Scalability:** The use of Python and appropriate libraries allows for potential scalability in handling multiple cameras and larger data volumes.
*   **Flexibility:** The camera interface is designed to accommodate various camera types, making the system adaptable to different retail settings.
* **Configuration**: The system configuration is managed via a json file and schema.

## Security and Data Privacy

*   **Data Handling:** Data is processed and stored securely.
*   **Access Control:** Access to sensitive data and system configurations is restricted.
* **Credentials**: User and credential management.
*   **Privacy Considerations:** The system adheres to privacy regulations by anonymizing customer data whenever possible.

## Future Considerations

*   Integration with other retail systems (e.g., POS, inventory management).
*   Advanced analytics using more sophisticated machine learning models.
*   Real-time alerts for abnormal customer behavior or high-traffic areas.
*   Cloud deployment for scalability and accessibility.

## Code Analysis and Recent Improvements

### Code Structure (Updated)

*   **Enhanced Modularity:** The codebase has been significantly improved with clean separation of concerns:
    *   `camera/` - Camera interfaces with unified factory pattern
    *   `detection/` - Person detection and tracking modules  
    *   `analysis/` - Customer behavior analysis modules
    *   `web/` - Web interface with clean service layer architecture
    *   `utils/` - Configuration and credential management utilities
*   **Service Layer Architecture:** Added `web/services.py` for business logic separation from routing
*   **Improved Configuration:** Centralized configuration management with secure credential handling
*   **Clean Entry Point:** `main.py` refactored with proper logging and path management
*   **Reduced Code Duplication:** Eliminated redundant files and duplicate functionality

### Recent Improvements Implemented

*   **Code Cleanup (2024 - Phase 2):**
    *   Removed duplicate files (`retail_analytics.db`, `retail_analytics.log`, `yolov8n.pt` in src/)
    *   Eliminated redundant `download_model.py` script
    *   Reduced `routes.py` from 1338 lines to 436 lines (68% reduction)
    *   Optimized `correlation.py` from 744 lines to ~400 lines (46% reduction)
    *   Extracted detection logic into reusable service layer
    *   Cleaned requirements.txt (removed duplicates and version conflicts)
    *   Fixed configuration file references to existing video files
    *   Removed IDE-specific files and empty directories
    *   Organized model files into proper directory structure
*   **Path Management:** Replaced all hard-coded paths with relative paths using `pathlib.Path`
*   **Logging Standardization:** Replaced print statements with proper logging throughout
*   **Configuration Centralization:** Improved config loading with environment awareness
*   **Error Handling:** Consistent error responses and comprehensive logging
*   **Service Architecture:** Clean separation between web routes and business logic

### Architecture Enhancements

*   **Web Layer Restructure:**
    *   `routes.py` - Clean route definitions organized by functionality
    *   `services.py` - Business logic for frame analysis and detection
    *   `__init__.py` - Simplified app factory with minimal route duplication
*   **Improved Maintainability:** Modular design makes testing and extension easier
*   **Better Debugging:** Structured logging with configurable levels throughout
*   **Cross-Platform Compatibility:** Proper path handling for different operating systems

### Performance Optimizations

*   **Reduced Memory Footprint:** Eliminated redundant data structures and duplicate code
*   **Optimized Imports:** Removed unnecessary imports and better dependency management
*   **Service Caching:** Intelligent caching in detection services to avoid redundant processing
*   **Efficient File Handling:** Better temporary file management in detection pipeline
*   **Code Complexity Reduction:** Extracted common patterns into reusable helper methods
*   **Dependency Optimization:** Cleaned requirements.txt to remove conflicts and duplicates
*   **File Organization:** Proper directory structure for models and configuration files

### Security and Reliability

*   **Credential Management:** Maintained secure credential system with encryption
*   **Error Recovery:** Graceful degradation when components fail
*   **Input Validation:** Consistent validation across all API endpoints
*   **Resource Cleanup:** Proper cleanup of temporary files and resources

### Testing Strategy

A comprehensive testing strategy is essential to ensure the reliability and performance of the retail analytics system. Here's a breakdown of how to approach different types of testing:

#### Unit Testing

*   **Purpose:** Verify that individual functions and classes work correctly in isolation.
*   **Tools:**
    *   `pytest`: A mature full-featured Python testing tool.
    *   `unittest`: Python's built-in unit testing framework.
*   **Methodology:**
    *   Write test cases for each module.
    *   Use mocks to simulate external dependencies (e.g., camera streams, ML models).
    *   Assert that function outputs match expected values for given inputs.
*   **Example:**
    *   Test camera interface functions to ensure they can connect to cameras and retrieve frames.
    *   Test analysis functions to verify they correctly calculate metrics like dwell time.

#### Integration Testing

*   **Purpose:** Verify that different modules work together as expected.
*   **Tools:** `pytest` or `unittest`.
*   **Methodology:**
    *   Test the interactions between modules (e.g., `camera` to `detection` to `analysis`).
    *   Simulate realistic scenarios, like processing a short video clip.
    *   Verify data flow and consistency between components.
*   **Example:**
    *   Test the full flow of capturing data from a camera, detecting objects, tracking them, and performing some analysis.

#### Performance Testing

*   **Purpose:** Evaluate the system's performance under different loads and identify bottlenecks.
*   **Tools:**
    *   `pytest-benchmark`: For benchmarking test execution times.
    *   `cProfile`: For profiling the execution of the program.
*   **Methodology:**
    *   Simulate various load conditions (e.g., multiple camera streams, high-traffic periods).
    *   Measure key performance indicators (e.g., frames per second, latency, memory usage).
    *   Identify performance bottlenecks.
*   **Example:**
    *   Test how many camera streams the system can process simultaneously without significant performance degradation.

#### ML Model Evaluation

*   **Purpose:** Evaluate the accuracy and efficiency of the machine learning models used for object detection and tracking.
*   **Tools:**
    *   `scikit-learn`: For calculating ML metrics.
    *   Custom scripts: To evaluate the accuracy.
*   **Methodology:**
    *   Use labeled datasets to validate the models.
    *   Calculate standard metrics (e.g., precision, recall, F1-score) to evaluate detection and tracking accuracy.
    *   Measure inference time for each model.
*   **Example:**
    *   Use a labeled dataset of video frames to assess the accuracy of the object detection model.

## Future Considerations

*   Integration with other retail systems (e.g., POS, inventory management).
*   Advanced analytics using more sophisticated/optimised machine learning models.
*   Real-time alerts for abnormal customer behavior or high-traffic areas.
*   Cloud deployment for scalability and accessibility.

# Retail Analytics System: Object Detection & Tracking Enhancements

## Overview
This document summarizes the enhanced object detection and tracking system implemented in the Retail Analytics platform. The primary focus was on reducing the "flashing" effect of bounding boxes and improving tracking stability while maintaining high confidence thresholds for accuracy.

## Key Enhancements

### Confidence Threshold Adjustments
- Increased base confidence threshold to 0.6 for reliable detections
- Implemented periodic lower threshold (0.4) every 30 frames to pick up new objects
- Improved IoU threshold to 0.75 for better non-maximum suppression
- Added confidence decay mechanism with a 50% maximum decay limit

### Tracking System Architecture
- Created a comprehensive tracking memory system with object persistence
- Added unique ID assignment and consistent tracking between frames
- Implemented ghost detection for objects that temporarily disappear
- Added frame count tracking for lifetime operations
- Implemented tracker initialization detection (stabilizes after 10 frames)

### Advanced Motion Tracking
- **Kalman Filter Integration**:
  - Added 4D state tracking (position and velocity)
  - Implemented physics-based motion model
  - Added configurable process and measurement noise parameters
  - Used for accurate trajectory prediction

- **Movement Classification**:
  - Added pattern classification (stationary, slow, fast)
  - Dynamically adjusted smoothing based on movement type
  - Applied stronger smoothing to stationary objects
  - Tracked movement parameters for better prediction

### Multi-level Smoothing
- Implemented position history tracking (last 5 frames)
- Added bounding box history for temporal averaging
- Applied exponential smoothing with configurable parameters:
  - Position smoothing: 0.7 (higher = smoother but less responsive)
  - Velocity smoothing: 0.85
  - Bounding box smoothing: 0.7
- Implemented dynamic smoothing based on movement speed
- Applied stronger smoothing for large movements

### Ghost Detection & Persistence
- Extended maximum ghost detection frames to 60 (2 seconds at 30fps)
- Implemented fade-in/fade-out effects for smooth transitions:
  - Fade-in duration: 10 frames
  - Fade-out duration: 20 frames
- Added motion prediction for disappeared objects
- Implemented velocity dampening that becomes gentler over time
- Applied time-based (dt) motion prediction

### UI/UX Enhancements
- Added fade level property for UI opacity control
- Included movement type with detections for visualization
- Added is_predicted flag to distinguish estimated positions
- Enhanced API response with tracking metadata
- Added stabilization level indicators

### Configuration Parameters
The system uses a centralized configuration in `detection_memory`:

```python
detection_memory = {
    'people': {},  # Store detected people by ID with tracking info
    'heatmap_points': [],  # Store heatmap data points
    'last_frame_time': None,  # Track time between frames
    'last_detections': [],  # Store last frame's detections for smoothing
    'detection_history': {},  # Track detection history for persistence
    'next_id': 1,  # ID counter for consistent tracking
    'tracker_initialized': False,  # Flag to track initialization status
    'min_tracking_confidence': 0.1,  # Minimum confidence to keep tracking
    'frame_count': 0,  # Count frames for cleanup operations
    'frame_interpolation': True,  # Enable interpolation between frames
    'max_ghost_frames': 60,  # Maximum frames to keep ghost detections (2 seconds at 30fps)
    'velocity_smoothing': 0.85,  # Higher values = smoother motion but less responsive
    'position_smoothing': 0.7,  # Higher values = smoother position but less responsive
    'bbox_smoothing': 0.7,  # Higher values = smoother boxes but less responsive
    'use_kalman_filter': KALMAN_AVAILABLE,  # Use Kalman filtering if available
    'multi_frame_average': 5,  # Number of frames to average for smoothing
    'fade_in_duration': 10,  # Frames to fade in new objects
    'fade_out_duration': 20,  # Frames to fade out disappearing objects
    'min_detection_confidence': 0.6,  # Minimum confidence for initial detection
    'process_noise': 0.01,  # Process noise for Kalman filter
    'measurement_noise': 0.1  # Measurement noise for Kalman filter
}
```

## Detection Pipeline

1. **Frame Capture**:
   - Decode base64 image data to OpenCV format
   - Extract frame dimensions and timestamp

2. **YOLO Detection**:
   - Run detection with appropriate confidence threshold
   - Process results to extract person detections
   - Convert bounding boxes to required format

3. **Tracking and Matching**:
   - Match new detections with previous frame objects
   - Calculate distance between detections for matching
   - Assign IDs to new detections or maintain existing ones

4. **Motion Analysis**:
   - Update Kalman filter with new measurements
   - Calculate velocity and predict motion
   - Classify movement patterns

5. **Position Smoothing**:
   - Apply multi-frame averaging to positions
   - Use Kalman filtered positions when available
   - Apply exponential smoothing to reduce jitter

6. **Bounding Box Smoothing**:
   - Apply temporal averaging to box dimensions
   - Apply stronger smoothing to size changes
   - Prevent rapid fluctuations in size

7. **Ghost Detection**:
   - Maintain objects that disappear temporarily
   - Predict positions for missing objects
   - Apply fade-out effect for smooth transitions

8. **Response Generation**:
   - Format detections with tracking metadata
   - Include analytics data (demographics, counts)
   - Provide tracking information for UI handling

## Dependencies

- **Ultralytics YOLO**: For object detection (`pip install ultralytics`)
- **FilterPy**: For Kalman filtering (`pip install filterpy`)
- **OpenCV**: For image processing
- **NumPy**: For numerical operations

## UI Integration Guidelines

For front-end developers integrating with this tracking system:

1. Use the `fade_level` property to control opacity of detection overlays:
   ```javascript
   const opacity = detection.fade_level || 1.0; 
   element.style.opacity = opacity;
   ```

2. Check for `is_predicted` flag to visually differentiate predicted vs detected positions:
   ```javascript
   if (detection.is_predicted) {
     // Use dashed border or different color
     element.classList.add('predicted');
   }
   ```

3. Use `movement_type` to apply different visual treatments:
   ```javascript
   element.classList.add(detection.movement_type || 'stationary');
   ```

4. Implement smooth transitions between positions using CSS:
   ```css
   .person {
     transition: all 0.15s ease-out;
   }
   ```

This enhanced tracking system significantly reduces bounding box flashing while maintaining high detection accuracy, providing a more stable and professional visualization of detected objects.