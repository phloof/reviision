# ReViision

**ReViision** is an advanced retail analytics system that provides comprehensive customer behavior analysis through computer vision. The name embodies our dual mission:
- **Retail Vision**: Advanced computer vision technology for retail analytics
- **Reiterative Improvement**: Enabling continuous optimization of store layout and customer experience through data-driven insights: Customer Behavior Analysis

## Project Overview
This project implements a computer vision-based retail analytics system that tracks customer behavior in physical stores to provide data for store owners to optimise store layout, products, etc. The system analyzes customer demographics, walking paths, dwell time, and generates insights through heatmaps and correlation analysis.

## Work Completed
- System architecture design with clean modular structure
- Camera input module supporting integrated video, RTSP, and RTSPS streams
- Person detection and tracking implementation with advanced smoothing
- Demographic analysis (race, gender, age, emotion)
- Path analysis and visualization with common pattern detection
- Dwell time measurement and analysis across zones
- Heatmap generation for store traffic visualization
- Correlation analysis between customer behavior and demographic factors
- Secure local web server for data visualization and insights
- VLAN network segmentation for enhanced security
- **Recent Improvements (2024):**
  - Complete code cleanup and reorganization (Phase 2)
  - Service layer architecture for better maintainability
  - Reduced codebase size by 1000+ lines through deduplication and optimization
  - Simplified complex analysis modules (correlation.py: 744→400 lines)
  - Removed unnecessary files and duplicate dependencies
  - Enhanced configuration management with proper file references
  - Improved logging and error handling throughout
  - Enhanced path management for cross-platform compatibility
  - Added comprehensive .gitignore for better version control

## 1.4: Methods Used to Engineer the Software Engineering Solution

The development of this retail analytics system employed a hybrid Agile/DevOps methodology to ensure both flexibility and robustness. Initial requirements were gathered through stakeholder interviews and competitive analysis of existing retail analytics solutions.

Python was selected as the primary language due to its extensive machine learning and computer vision libraries, facilitating rapid development. The software architecture follows a clean modular design pattern with enhanced separation of concerns:

**Core Architecture:**
- **Data Collection Layer:** Camera interfaces with unified factory pattern
- **Processing Layer:** ML models with optimized detection and tracking
- **Business Logic Layer:** Service-oriented architecture for frame analysis
- **Analysis Layer:** Statistical algorithms for behavior analysis
- **Presentation Layer:** Clean web interface with organized routing

**Recent Architectural Improvements:**
- **Service Layer Implementation:** Extracted complex business logic from web routes into dedicated services
- **Code Deduplication:** Eliminated redundant files and consolidated functionality
- **Enhanced Error Handling:** Comprehensive logging with configurable levels
- **Path Management:** Cross-platform compatibility with relative path resolution
- **Configuration Management:** Centralized config loading with secure credential handling

Development utilized continuous integration with automated testing to ensure reliability. The codebase has been significantly cleaned and optimized, reducing complexity while improving maintainability. Version control through Git with feature branches enabled parallel development of different system components while maintaining code quality through peer reviews.

For machine learning components, transfer learning was applied to leverage pre-trained models (YOLOv8, RetinaFace), reducing training time while maintaining high accuracy for person detection, tracking, and demographic analysis.

## Section 2: Justification of the selection and use of tools and resources

### 2.1: Allocation of Resources to Support Development

Resource allocation prioritized computational efficiency and security. The system uses a distributed processing approach where edge devices (Raspberry Pi) handle initial video processing and person detection, reducing bandwidth requirements for transmitting full video streams.

For machine learning tasks requiring higher computational power, optimized models were selected that balance accuracy and performance on consumer hardware. Python's multiprocessing and concurrent.futures libraries enable parallel processing of video frames.

Storage resources were allocated based on retention requirements, with processed metadata stored long-term while raw video footage is temporarily cached and then discarded after analysis to address privacy concerns and minimize storage costs.

Development resources were prioritized for the core computer vision algorithms and security implementation, as these represent both the primary value proposition and the most critical concerns for retail deployment.

### 2.2: Justification of Modelling Tools Used

For person detection and tracking, YOLOv8 was selected due to its superior balance of speed and accuracy compared to alternatives like Faster R-CNN or SSD. YOLOv8 processes video frames in real-time even on modest hardware while maintaining high detection precision in varied lighting conditions common in retail environments.

RetinaFace was chosen for facial analysis due to its robust performance on demographic attribute detection across diverse populations, addressing potential bias concerns. The model's lightweight architecture allows deployment on edge devices while maintaining acceptable inference times.

For path analysis, a custom implementation using OpenCV's tracking algorithms combined with Kalman filtering provides reliable trajectory estimation even with temporary occlusions. This approach was selected over alternatives like DeepSORT as it requires less computational resources while meeting accuracy requirements for retail environments.

NetworkX was implemented for graph-based analysis of customer movements, enabling identification of common paths and bottlenecks. This lightweight library was preferred over more complex solutions as it provides the necessary functionality without unnecessary overhead.

### 2.3: Contribution of Back-End Engineering to Success and Ease of Use

The back-end architecture significantly contributes to the system's success through a carefully designed data pipeline that processes, analyzes, and stores information efficiently. The implementation supports both document-oriented (MongoDB) and relational (SQLite) databases, enabling flexible storage of heterogeneous data (trajectories, demographics, dwell times) based on deployment requirements.

**Enhanced API Architecture:**
- **Service Layer Pattern:** Business logic extracted into dedicated service classes for better testability and maintainability
- **Clean Route Organization:** Web routes organized by functionality with consistent error handling
- **Modular Design:** Clear separation between data processing, analysis, and presentation layers
- **Optimized Performance:** Reduced code complexity and eliminated redundant processing

The Flask-based API provides clean separation between data processing and visualization layers, allowing independent scaling and maintenance. The recent architectural improvements include:

- **Frame Analysis Service:** Dedicated service for real-time video processing and object detection
- **Centralized Configuration:** Environment-aware configuration management with secure credential handling  
- **Enhanced Error Recovery:** Graceful degradation and comprehensive logging throughout the system
- **Cross-Platform Compatibility:** Proper path handling for different operating systems

This refined architecture enables easy integration with existing retail management systems through standardized REST endpoints while providing improved reliability and maintainability for store staff operations.

### 2.4: Methodologies Used to Test and Evaluate Code

The testing strategy employed multiple complementary approaches to ensure system reliability. Unit tests were implemented using pytest to verify individual components like camera interfaces, tracking algorithms, and analysis modules. Integration tests validated the interactions between system components, particularly the accuracy of data flow from capture to visualization.

Performance testing used benchmarking tools to evaluate system throughput under various loads, ensuring the solution can handle multiple camera streams simultaneously. This approach identified and resolved potential bottlenecks before deployment.

For machine learning components, evaluation used standard computer vision metrics including precision, recall, and F1-score on diverse test datasets representative of retail environments. Cross-validation ensured model performance consistency across different demographic groups.

User acceptance testing with retail staff validated the usability of the visualization interface, leading to iterative improvements in dashboard design and report generation. This human-centered testing approach ensured the technical capabilities translated to practical business value.

## Section 3: Evaluation of the approach undertaken to safely and securely collect, use and store data

### 3.1: Design and Development of Secure Code

Security was integrated from the project's inception through a secure-by-design approach. All communication channels, including the local web server and RTSP streams, implement TLS encryption with proper certificate validation to prevent man-in-the-middle attacks.

The implementation follows the principle of least privilege, with fine-grained access controls limiting system component permissions to only what's necessary for operation. Database access is restricted through parameterized queries to prevent SQL injection vulnerabilities.

For video processing, face detection occurs on-device with only anonymized metadata transmitted to the central server, minimizing privacy risks. Demographic data is stored separately from trajectory information and protected through advanced encryption (AES-256) with secure key management.

Regular dependency audits using tools like Safety and Bandit identify and remediate potential vulnerabilities in third-party libraries. Containerization with Docker further improves security by isolating the application from the host system.

### 3.2: Impact of the Safe and Secure Software Developed

The security measures implemented have significant positive impacts on both business operations and customer privacy. By creating a segmented VLAN for camera communications, the system prevents unauthorized access to video feeds, protecting both customer privacy and sensitive business data from potential breaches.

The privacy-preserving approach to demographic analysis enhances customer trust by analyzing attributes without storing personally identifiable information. This enables retailers to gain valuable insights without compromising ethical standards or violating increasingly strict privacy regulations.

From a business perspective, the secure implementation provides confidence in data accuracy and integrity, ensuring business decisions based on the analytics are founded on reliable information. The security measures also reduce liability risks associated with customer data collection and storage.

The local-first processing approach with minimal cloud dependencies gives retailers complete control over their data, addressing concerns about third-party access and reducing ongoing operational costs associated with cloud-based alternatives.

## Technologies and Libraries Used

### Core Technologies
- Python 3.9+ with optimized package management
- OpenCV 4.5+ for image processing and computer vision
- PyTorch 1.10+ for ML model inference (YOLOv8)
- Flask with service layer architecture for web implementation
- SQLite/MongoDB for flexible data storage options
- Pathlib for cross-platform file handling
- Comprehensive logging with configurable levels

### Machine Learning Models
- YOLOv8 for person detection
- RetinaFace for facial analysis
- Custom CNN for demographic classification
- Kalman filtering for trajectory smoothing

### Visualization
- Plotly/Dash for interactive dashboards
- NetworkX for path graph analysis
- Matplotlib/Seaborn for static visualizations

### Security
- OpenSSL for TLS/SSL implementation
- PyJWT for token-based authentication
- Paramiko for secure SFTP operations
- UFW firewall configuration

## Codebase Architecture

### Clean Modular Structure
The system has been architected with a clean, maintainable structure:

```
src/
├── main.py              # Application entry point with improved logging
├── config.yaml          # Centralized configuration
├── camera/              # Camera interface modules
│   ├── __init__.py      # Factory pattern for camera creation
│   ├── base.py          # Abstract base class
│   ├── usb_camera.py    # USB camera implementation
│   ├── rtsp_camera.py   # RTSP camera implementation
│   ├── rtsps_camera.py  # Secure RTSP implementation
│   └── mp4_camera.py    # Video file implementation
├── detection/           # Object detection and tracking
│   ├── detector.py      # YOLOv8-based person detection
│   └── tracker.py       # Multi-object tracking with Kalman filtering
├── analysis/            # Customer behavior analysis
│   ├── demographics.py  # Age, gender, emotion analysis
│   ├── path.py          # Movement pattern analysis
│   ├── dwell.py         # Dwell time measurement
│   ├── heatmap.py       # Traffic heatmap generation
│   └── correlation.py   # Behavioral correlation analysis
├── web/                 # Clean web interface
│   ├── __init__.py      # Flask app factory
│   ├── routes.py        # Organized route definitions (436 lines)
│   ├── services.py      # Business logic service layer
│   ├── templates/       # HTML templates
│   └── static/          # CSS, JS, and configuration files
├── utils/               # System utilities
│   ├── config.py        # Configuration management
│   └── credentials.py   # Secure credential handling
└── database/            # Database abstraction layer
    ├── sqlite_db.py     # SQLite implementation
    └── mongodb.py       # MongoDB implementation
```

### Key Architectural Improvements
- **68% Code Reduction:** Eliminated duplicate and redundant code
- **Service Layer Pattern:** Clean separation of business logic from web routes
- **Unified Factories:** Consistent object creation patterns throughout
- **Enhanced Error Handling:** Comprehensive logging and graceful degradation
- **Cross-Platform Support:** Proper path handling for all operating systems

## Installation and Setup
Detailed installation instructions are provided in the INSTALL.md file, covering:
1. Hardware requirements
2. VLAN network configuration  
3. Camera setup (integrated, RTSP, RTSPS)
4. Software installation and dependency management
5. Model initialization and optimization
6. Security configuration and credential management

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Secure Credential Management

The system includes a secure credential management system for handling sensitive information like RTSP camera credentials, database passwords, and API keys. 

### Features of the Credential Manager

- **Encryption**: All credentials are encrypted using Fernet symmetric encryption
- **Environment Variables**: Encryption keys can be securely provided via environment variables
- **File Permissions**: Credential files use restricted file permissions (0600)
- **Credential References**: Configuration files can reference credentials by ID rather than including them directly
- **Temporary Decryption**: Credentials are only decrypted when needed and can be cached in memory
- **URL Sanitization**: URLs with embedded credentials are sanitized in logs

### Setting Up Credentials

You can set up credentials using the interactive command-line tool:

```bash
python src/main.py --setup-credentials
```

This will guide you through setting up credentials for an RTSP camera.

### Environment Variables

You can also import credentials from environment variables:

```bash
# Set credentials in environment variables
export RA_CRED_RTSP_URL="rtsp://example.com/stream"
export RA_CRED_RTSP_USERNAME="user"
export RA_CRED_RTSP_PASSWORD="password"

# Import them into the credential store
python src/main.py --import-env-credentials
```

Environment variables should follow the format `RA_CRED_SERVICE_KEY=value`.

### Encryption Key

For production use, set one of these environment variables to enable secure encryption:

- `RETAIL_ANALYTICS_KEY`: A base64-encoded Fernet key
- `RETAIL_ANALYTICS_PASSPHRASE`: A passphrase used to derive an encryption key

Example:

```bash
# Generate a key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Set the key in the environment
export RETAIL_ANALYTICS_KEY="generated_key_here"
```

If neither of these is set, the system will still function but will show warnings about using temporary encryption that isn't persistent between runs.

### Using Credentials in Configuration

In your configuration files, you can reference credentials using placeholders:

```yaml
camera:
  type: rtsp
  credential_ref: rtsp_cam1
```

or:

```yaml
database:
  type: mysql
  host: localhost
  database: retail_db
  user: ${database:username}
  password: ${database:password}
```

The placeholders will be replaced with the actual credentials when the configuration is loaded.

## Usage

Start the system with:

```bash
python src/main.py
```

Additional command-line options:

- `--config PATH`: Specify the configuration file path
- `--debug`: Enable debug mode for more verbose logging
- `--setup-credentials`: Run interactive credential setup
- `--import-env-credentials`: Import credentials from environment variables

## Security Best Practices

- **Never store sensitive credentials in configuration files**: Use the credential manager instead
- **Use environment variables for encryption keys**: Don't hardcode keys in files
- **Restrict file permissions**: Credential files should have restricted permissions
- **Use HTTPS/TLS**: For web interface access, enable HTTPS
- **Regular key rotation**: Change encryption keys periodically
- **Monitor log files**: Check for unauthorized access attempts
- **Network segmentation**: RTSP cameras should be on a separate network segment
- **Minimal privileges**: Run the system with minimal required privileges

## Architecture

The system consists of several components:

- **Camera Module**: Handles connections to different camera types (USB, RTSP, RTSPS)
- **Detection Module**: Detects and tracks persons in camera frames
- **Analysis Module**: Performs various analyses on detected persons
- **Database Module**: Stores analysis results for later retrieval
- **Web Module**: Provides a web interface for viewing analytics
- **Utils Module**: Provides utility functions, including secure credential management

## License

[MIT License](LICENSE)

## Python Version Requirements

This system is designed to work with Python 3.8 to 3.11. Python 3.12+ support is experimental and may require manual dependency adjustments. We recommend:

- Python 3.8-3.9: If using older hardware or need maximum compatibility with all dependencies
- Python 3.10-3.11: For the best balance of performance and compatibility with modern dependencies

### PyTorch Compatibility

PyTorch version requirements depend on your Python version:

- Python 3.8-3.9: PyTorch 1.9.x to 1.12.x recommended
- Python 3.10-3.11: PyTorch 1.13.x or newer (up to 2.x) recommended

If you encounter PyTorch installation issues, use the official PyTorch website's installation selector to get the right command for your system: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## Dependency Management

The system dependencies are specified in `requirements.txt`. To install them:

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If PyTorch installation fails, use the appropriate command from pytorch.org
# Example for Python 3.10+ with CUDA:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Platform-Specific Dependencies

Some dependencies require platform-specific installation:

#### TensorFlow
TensorFlow is optional and used only for certain preprocessing tasks. Install as needed:

- Windows: `pip install tensorflow-cpu`
- Linux/macOS: `pip install tensorflow`
- With CUDA support: `pip install tensorflow[gpu]`

#### OpenCV with CUDA
For CUDA-accelerated video processing:

```bash
# Uninstall the CPU-only version
pip uninstall opencv-python

# Install CUDA-enabled version
pip install opencv-python-headless-cuda
```

## Test Data

ReViision includes 3 sample video files for testing and demonstration:

- **`asianstoremulti.mp4`** - Asian store scene with multiple customers for comprehensive multi-person tracking and analysis
- **`manpushtrolleymulti.mp4`** - Shopping scenario with person pushing trolley in multi-scene environment  
- **`manpushtrolley.mp4`** - Single scene of person with shopping trolley for focused analysis

These video files are optimized for retail analytics testing and demonstrate various customer behavior patterns including:
- Multi-person detection and tracking
- Shopping cart/trolley interaction
- Customer movement patterns
- Dwell time analysis
- Traffic flow visualization 