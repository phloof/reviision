# Retail Analytics System - Project Logbook

## Week 6, Term 1 (03/03/2025-08/03/2025)

### Planning
**Milestones:**
Initial camera interface setup and RTSP research, begin YOLOv8 integration, establish project structure

**Planned:**
1.1 Research RTSP protocol and authentication methods
1.2 Design camera interface architecture 
1.3 Set up YOLOv8 environment and dependencies
1.4 Create basic project directory structure
1.5 Initialise Git repository and version control

**Actual:**
1.1 - 03/03/2025 External work
1.2 - 04/03/2025 Class Time
1.3 - 05/03/2025 External work  
1.3 - 06/03/2025 External work
1.4 - 07/03/2025 External work
1.5 - 08/03/2025 External work

**Comment:**
Research showed RTSP authentication varies significantly between manufacturers. Started basic camera interface design. YOLOv8 installation successful after resolving dependency conflicts. Project structure follows standard Python package layout.

### Issues
**Description:**
RTSP authentication complexity across different camera brands
YOLOv8 dependency conflicts with existing OpenCV installation

**Dates:**
03/03/2025 External work
05/03/2025 External work

**Action/Results:**
Researched manufacturer documentation for authentication protocols
Created separate virtual environment for YOLOv8 dependencies  

**Finished (Y/N):**
Yes
Yes

---

## Week 7, Term 1 (09/03/2025-15/03/2025)

### Planning
**Milestones:**
Complete camera interfaces for USB and MP4, implement basic person detection, start tracking module

**Planned:**
1.1 Implement USB camera interface using OpenCV
1.2 Create MP4 file reader for testing
1.3 Test YOLOv8 person detection on sample footage
1.4 Research tracking algorithms for retail environments
1.5 Begin Kalman filter implementation for trajectory smoothing

**Actual:**
1.1 - 10/03/2025 External work
1.2 - 11/03/2025 Class Time
1.3 - 12/03/2025 External work
1.3 - 13/03/2025 External work
1.4 - 14/03/2025 External work
1.5 - 15/03/2025 External work

**Comment:**

### Issues
**Description:**
USB camera frame rate inconsistent on different hardware
Tracking ID conflicts when people temporarily leave camera view

**Dates:**
10/03/2025 External work
13/03/2025 External work

**Action/Results:**
Added adaptive frame rate detection and adjustment
Implemented ID persistence with timeout mechanism

**Finished (Y/N):**
Yes
Yes

---

## Week 8, Term 1 (16/03/2025-22/03/2025)

### Planning
**Milestones:**
Implement demographic analysis, create path analysis module, design dwell time calculation

**Planned:**
1.1 Integrate RetinaFace for facial analysis
1.2 Add age and gender classification models
1.3 Design path analysis using graph theory
1.4 Implement zone-based dwell time calculation
1.5 Create basic Flask web interface structure

**Actual:**
1.1 - 17/03/2025 External work
1.2 - 18/03/2025 Class Time
1.3 - 19/03/2025 External work
1.4 - 20/03/2025 External work
1.4 - 21/03/2025 External work
1.5 - 22/03/2025 External work

**Comment:**
RetinaFace integration successful with good accuracy. Age/gender models showing acceptable performance. NetworkX chosen for path analysis. Zone configuration system implemented. Basic Flask structure established.

### Issues
**Description:**
RetinaFace memory usage high on consumer hardware

**Dates:**
17/03/2025 External work

**Action/Results:**
Optimised RetinaFace inference with batch processing

**Finished (Y/N):**
Yes

---

## Week 9, Term 1 (23/03/2025-29/03/2025)

### Planning
**Milestones:**
Complete heatmap generation, implement correlation analysis, enhance web dashboard

**Planned:**
1.1 Design traffic heatmap visualisation algorithms
1.2 Implement statistical correlation analysis framework
1.3 Create dashboard templates and routing
1.4 Add real-time data streaming capability
1.5 Implement secure credential management system

**Actual:**
1.1 - 24/03/2025 External work
1.2 - 25/03/2025 Class Time
1.3 - 26/03/2025 External work
1.4 - 27/03/2025 External work
1.5 - 28/03/2025 External work
1.5 - 29/03/2025 External work

**Comment:**
Heatmap generation produces clear traffic visualisation. Correlation analysis reveals interesting behaviour patterns. Dashboard responsive design implemented. WebSocket streaming working for real-time updates. Credential encryption using AES-256.

### Issues
**Description:**
Heatmap rendering slow on high-resolution displays
Correlation calculations expensive with large datasets
Real-time updates causing browser performance issues

**Dates:**
24/03/2025 External work
27/03/2025 External work
29/03/2025 External work

**Action/Results:**
Implemented multi-resolution heatmap with dynamic scaling
Added caching and parallel processing for correlations
Optimised WebSocket data transmission with compression

**Finished (Y/N):**
Yes
Yes
Partial

---

## Week 10, Term 1 (30/03/2025-05/04/2025)

### Planning
**Milestones:**
System testing and optimisation, documentation, Term 1 completion

**Planned:**
1.1 Create unit tests for core modules
1.2 Implement integration testing pipeline
1.3 Profile and optimise system performance
1.4 Write API documentation and user guides
1.5 Complete Term 1 development milestones

**Actual:**
1.1 - 31/03/2025 External work
1.2 - 01/04/2025 External work
1.3 - 02/04/2025 External work
1.4 - 04/04/2025 Class Time
1.3 - 04/04/2025 External work
1.5 - 05/04/2025 External work

**Comment:**
Unit testing revealed several edge cases. Integration tests validate end-to-end workflow. Performance profiling identified memory leak in detection loop. Documentation following industry standards. Term 1 development objectives successfully completed.

### Issues
**Description:**
Memory leaks in long-running detection processes

**Dates:**
04/04/2025 Class Time

**Action/Results:**
Fixed memory management with proper resource cleanup

**Finished (Y/N):**
Yes

---

## Week 11, Term 1 (06/04/2025-12/04/2025)

### Planning
**Milestones:**
Security implementation, configuration management, Term 1 completion

**Planned:**
1.1 Implement VLAN network segmentation design
1.2 Add TLS encryption for all communications
1.3 Create configuration validation system
1.4 Enhance logging and monitoring
1.5 Complete security and configuration implementation

**Actual:**
1.1 - 07/04/2025 Class Time
1.2 - 08/04/2025 External work
1.3 - 09/04/2025 External work
1.4 - 10/04/2025 External work
1.4 - 11/04/2025 External work
1.5 - 12/04/2025 External work

**Comment:**
VLAN configuration documented for camera isolation. TLS implementation secures all endpoints. JSON schema validation prevents configuration errors. Structured logging improves debugging. Security and configuration systems fully implemented.

### Issues
**Description:**
VLAN setup complexity varies across network equipment
Log volume impacts performance in debug mode

**Dates:**
07/04/2025 Class Time
11/04/2025 External work

**Action/Results:**
Created vendor-specific network configuration guides
Implemented intelligent log rotation and level management

**Finished (Y/N):**
Yes
Yes

---

## Week 1, Term 2 (27/04/2025-03/05/2025)

### Planning
**Milestones:**
Complete core detection system, begin web interface development

**Planned:**
1.1 Finalise person detection and tracking accuracy
1.2 Complete demographic analysis integration
1.3 Start basic Flask web interface design
1.4 Design REST API endpoints for data access
1.5 Plan web dashboard architecture

**Actual:**
1.1 - 28/04/2025 External work
1.2 - 29/04/2025 External work
1.3 - 30/04/2025 External work
1.4 - 01/05/2025 External work
1.5 - 02/05/2025 External work
1.5 - 03/05/2025 Class Time

**Comment:**
Detection accuracy now consistently above 90%. Demographic analysis working with RetinaFace integration. Basic Flask routes established for web interface. API endpoint design completed. Dashboard wireframes ready for implementation.

### Issues
**Description:**
Flask routing structure needs better organisation
API response format standardisation required

**Dates:**
30/04/2025 External work
02/05/2025 External work

**Action/Results:**
Created blueprints for better route organisation
Implemented consistent JSON response format

**Finished (Y/N):**
Yes
Yes

---

## Week 2, Term 2 (04/05/2025-10/05/2025)

### Planning
**Milestones:**
Web dashboard implementation, real-time data streaming

**Planned:**
1.1 Implement dashboard HTML templates
1.2 Add JavaScript for real-time updates
1.3 Create data visualisation charts
1.4 Implement WebSocket for live streaming
1.5 Add basic authentication system

**Actual:**
1.1 - 05/05/2025 Class Time
1.2 - 06/05/2025 External work
1.3 - 07/05/2025 External work
1.4 - 08/05/2025 External work
1.4 - 09/05/2025 External work
1.5 - 10/05/2025 External work

**Comment:**
Dashboard templates responsive and functional. JavaScript successfully handles real-time data updates. Charts display heatmaps and traffic patterns clearly. WebSocket connection stable for live streaming. Basic login system protecting sensitive data.

### Issues
**Description:**
WebSocket connections dropping intermittently
Chart rendering slow with large datasets

**Dates:**
07/05/2025 External work
09/05/2025 External work

**Action/Results:**
Added connection retry logic and heartbeat mechanism
Implemented data sampling for improved chart performance

**Finished (Y/N):**
Yes
Yes

---

## Week 3, Term 2 (11/05/2025-17/05/2025)

### Planning
**Milestones:**
Advanced analytics features, user interface enhancements

**Planned:**
1.1 Implement heatmap generation and visualisation
1.2 Add correlation analysis between customer patterns
1.3 Create settings page for configuration management
1.4 Implement data export functionality
1.5 Add user management and roles system

**Actual:**
1.1 - 12/05/2025 External work
1.2 - 13/05/2025 Class Time
1.3 - 14/05/2025 External work
1.4 - 15/05/2025 External work
1.4 - 16/05/2025 External work
1.5 - 17/05/2025 External work

**Comment:**
Heatmap visualisation showing clear traffic patterns. Correlation analysis reveals interesting customer behaviour insights. Settings page allows dynamic configuration updates. Export functionality generates CSV and JSON reports. User roles system controls access to sensitive features.

### Issues
**Description:**
Heatmap calculation intensive for real-time display
User authentication session management problems

**Dates:**
14/05/2025 External work
16/05/2025 External work

**Action/Results:**
Implemented background heatmap processing with caching
Fixed session timeout and cookie security issues

**Finished (Y/N):**
Yes
Yes

---

## Week 4, Term 2 (18/05/2025-24/05/2025)

### Planning
**Milestones:**
Database integration, historical data analysis, security enhancements

**Planned:**
1.1 Implement MongoDB integration for data storage
1.2 Create historical data analysis views
1.3 Add HTTPS and SSL certificate configuration
1.4 Implement data backup and recovery systems
1.5 Create comprehensive logging and monitoring

**Actual:**
1.1 - 19/05/2025 Class Time
1.2 - 20/05/2025 External work
1.3 - 21/05/2025 External work
1.4 - 22/05/2025 External work
1.4 - 23/05/2025 External work
1.5 - 24/05/2025 External work

**Comment:**
MongoDB successfully storing detection data and analytics. Historical views showing trends over time periods. HTTPS working with self-signed certificates for development. Backup system creating daily snapshots. Comprehensive logging tracking all system events.

### Issues
**Description:**
MongoDB connection timeouts under heavy load
SSL certificate configuration complex for deployment

**Dates:**
20/05/2025 External work
23/05/2025 External work

**Action/Results:**
Implemented connection pooling and retry logic
Created automated certificate generation scripts

**Finished (Y/N):**
Yes
Yes

---

## Week 5, Term 2 (25/05/2025-31/05/2025)

### Planning
**Milestones:**
Testing, documentation, deployment preparation

**Planned:**
1.1 Comprehensive system testing and bug fixes
1.2 Create user documentation and guides
1.3 Implement automated deployment scripts
1.4 Performance testing and optimisation
1.5 Security testing and vulnerability assessment

**Actual:**
1.1 - 26/05/2025 Class Time
1.2 - 27/05/2025 External work
1.3 - 28/05/2025 External work
1.4 - 29/05/2025 External work
1.4 - 30/05/2025 External work
1.5 - 31/05/2025 External work

**Comment:**
System testing revealed several minor bugs now fixed. User documentation covers installation and operation procedures. Deployment scripts automate server setup process. Performance optimised for multiple concurrent users. Security assessment passed with minor recommendations implemented.

### Issues
**Description:**
Load testing revealed memory usage spikes
Session management vulnerabilities identified

**Dates:**
27/05/2025 External work
28/05/2025 External work

**Action/Results:**
Implemented garbage collection and memory pooling
Enhanced session security with timeout and rotation

**Finished (Y/N):**
Yes
Yes

---

## Week 6, Term 2 (01/06/2025-07/06/2025)

### Planning
**Milestones:**
Code organisation and cleanup Phase 1

**Planned:**
1.1 Reorganise project file structure and directories
1.2 Remove duplicate code and unused functions
1.3 Standardise coding style and formatting
1.4 Update import statements and dependencies
1.5 Create modular service architecture

**Actual:**
1.1 - 02/06/2025 External work
1.2 - 03/06/2025 Class Time
1.3 - 04/06/2025 External work
1.4 - 05/06/2025 External work
1.4 - 06/06/2025 External work
1.5 - 07/06/2025 External work

**Comment:**
Project structure reorganised with logical module separation. Removed significant amount of duplicate code across files. Applied consistent PEP 8 formatting throughout codebase. Dependencies cleaned up in requirements.txt. Service layer architecture improves code maintainability.

### Issues
**Description:**
Refactoring broke some existing imports
Large files difficult to reorganise safely

**Dates:**
04/06/2025 External work
06/06/2025 External work

**Action/Results:**
Fixed import paths systematically across all modules
Used incremental refactoring approach for large files

**Finished (Y/N):**
Yes
Yes

---

## Week 7, Term 2 (08/06/2025-14/06/2025)

### Planning
**Milestones:**
Code optimisation and cleanup Phase 2

**Planned:**
1.1 Optimise correlation analysis algorithms
1.2 Reduce web routes complexity significantly
1.3 Implement proper error handling throughout
1.4 Add comprehensive logging and monitoring
1.5 Create unit tests for core functionality

**Actual:**
1.1 - 09/06/2025 External work
1.2 - 10/06/2025 External work
1.3 - 11/06/2025 External work
1.4 - 12/06/2025 External work
1.5 - 13/06/2025 Class Time
1.5 - 14/06/2025 External work

**Comment:**
Correlation analysis module reduced from 744 to 400 lines with better performance. Web routes streamlined from 1338 to 436 lines eliminating redundancy. Robust error handling prevents system crashes. Structured logging provides detailed system insights. Unit test coverage now over 80%.

### Issues
**Description:**
Algorithm optimisation breaking backward compatibility
Test coverage gaps in optimised code sections

**Dates:**
10/06/2025 External work
12/06/2025 External work

**Action/Results:**
Maintained API compatibility while improving performance
Added comprehensive tests for all optimised functions

**Finished (Y/N):**
Yes
Yes

---

## Week 8, Term 2 (15/06/2025-21/06/2025)

### Planning
**Milestones:**
Final website implementation and polish

**Planned:**
1.1 Complete remaining dashboard pages and features
1.2 Implement responsive design for mobile devices
1.3 Add advanced filtering and search functionality
1.4 Create comprehensive admin panel
1.5 Finalise user interface styling and branding

**Actual:**
1.1 - 16/06/2025 External work
1.2 - 17/06/2025 Class Time
1.3 - 18/06/2025 External work
1.4 - 19/06/2025 External work
1.4 - 20/06/2025 External work
1.5 - 21/06/2025 External work

**Comment:**
All dashboard pages now functional with consistent design. Mobile responsive layout working across devices. Advanced search allows filtering by multiple criteria. Admin panel provides full system control and monitoring. Professional styling with consistent branding applied throughout.

### Issues
**Description:**
Mobile layout breaking on smaller screen sizes
Admin panel permissions not properly restricting access

**Dates:**
18/06/2025 External work
20/06/2025 External work

**Action/Results:**
Fixed CSS media queries for proper mobile rendering
Implemented role-based access control with proper validation

**Finished (Y/N):**
Yes
Yes

---

## Week 9, Term 2 (22/06/2025-28/06/2025)

### Planning
**Milestones:**
Advanced website features and system integration

**Planned:**
1.1 Implement advanced analytics dashboard features
1.2 Add data export and reporting functionality
1.3 Complete system integration testing
1.4 Optimise database queries and performance
1.5 Implement automated backup and recovery systems

**Actual:**
1.1 - 23/06/2025 Class Time
1.2 - 24/06/2025 External work
1.3 - 25/06/2025 External work
1.4 - 26/06/2025 External work
1.4 - 27/06/2025 External work
1.5 - 28/06/2025 External work

**Comment:**
Advanced dashboard features including trend analysis and predictive insights implemented. Data export functionality generates comprehensive reports in multiple formats. Integration testing identifies and resolves several compatibility issues. Database query optimisation improves response times significantly. Automated backup system ensures data security and recovery capabilities.

### Issues
**Description:**
Complex analytics calculations causing performance bottlenecks
Database backup process interfering with live operations

**Dates:**
24/06/2025 External work
27/06/2025 External work

**Action/Results:**
Implemented asynchronous processing for heavy calculations
Scheduled backups during low-traffic periods with minimal impact

**Finished (Y/N):**
Yes
Yes

---

## Week 10, Term 2 (29/06/2025-05/07/2025)
