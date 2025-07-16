# ReViision Database Schema Documentation

## Overview
This document describes the database schema for the ReViision Retail Analytics System. The schema has been designed to comply with Third Normal Form (3NF) to ensure data integrity, eliminate redundancy, and maintain efficient data relationships.

## Database Schema Normalization

### Third Normal Form (3NF) Requirements
1. **First Normal Form (1NF)**: Each table cell contains only atomic values, and each record is unique
2. **Second Normal Form (2NF)**: All non-key attributes are fully functionally dependent on the primary key
3. **Third Normal Form (3NF)**: No transitive dependencies - non-key attributes should not depend on other non-key attributes

## Current Schema Issues and Fixes

### Issues Identified:
1. Missing `dwell_times` table (referenced in queries but not created)
2. Age groups calculated in application code instead of normalized lookup
3. Bounding box coordinates stored redundantly without proper normalization
4. No foreign key constraints defined
5. Calculated fields done at query time instead of proper normalization

## New 3NF-Compliant Schema

### Core Entities

#### 1. Persons Table
Stores unique individuals detected by the system.
```sql
CREATE TABLE persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_detected DATETIME NOT NULL,
    last_detected DATETIME NOT NULL,
    total_visits INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 2. Age Groups Table (Lookup)
Normalized age group classifications.
```sql
CREATE TABLE age_groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_name VARCHAR(20) NOT NULL UNIQUE,
    min_age INTEGER NOT NULL,
    max_age INTEGER NOT NULL,
    display_order INTEGER NOT NULL
);

-- Default data:
INSERT INTO age_groups (group_name, min_age, max_age, display_order) VALUES
    ('Under 18', 0, 17, 1),
    ('18-24', 18, 24, 2),
    ('25-34', 25, 34, 3),
    ('35-44', 35, 44, 4),
    ('45-54', 45, 54, 5),
    ('55-64', 55, 64, 6),
    ('65+', 65, 120, 7);
```

#### 3. Genders Table (Lookup)
Normalized gender classifications.
```sql
CREATE TABLE genders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gender_name VARCHAR(20) NOT NULL UNIQUE,
    display_name VARCHAR(20) NOT NULL,
    icon_class VARCHAR(50)
);

-- Default data:
INSERT INTO genders (gender_name, display_name, icon_class) VALUES
    ('male', 'Male', 'fas fa-mars'),
    ('female', 'Female', 'fas fa-venus'),
    ('unknown', 'Unknown', 'fas fa-question');
```

#### 4. Emotions Table (Lookup)
Normalized emotion classifications.
```sql
CREATE TABLE emotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    emotion_name VARCHAR(20) NOT NULL UNIQUE,
    display_name VARCHAR(20) NOT NULL,
    icon_class VARCHAR(50),
    color_class VARCHAR(20)
);

-- Default data:
INSERT INTO emotions (emotion_name, display_name, icon_class, color_class) VALUES
    ('happy', 'Happy', 'fas fa-smile', 'success'),
    ('sad', 'Sad', 'fas fa-frown', 'info'),
    ('angry', 'Angry', 'fas fa-angry', 'danger'),
    ('surprised', 'Surprised', 'fas fa-surprise', 'warning'),
    ('neutral', 'Neutral', 'fas fa-meh', 'secondary'),
    ('focused', 'Focused', 'fas fa-eye', 'primary'),
    ('tired', 'Tired', 'fas fa-tired', 'dark');
```

#### 5. Races Table (Lookup)
Normalized race/ethnicity classifications.
```sql
CREATE TABLE races (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    race_name VARCHAR(30) NOT NULL UNIQUE,
    display_name VARCHAR(30) NOT NULL
);

-- Default data:
INSERT INTO races (race_name, display_name) VALUES
    ('white', 'White'),
    ('black', 'Black'),
    ('asian', 'Asian'),
    ('hispanic', 'Hispanic'),
    ('other', 'Other'),
    ('unknown', 'Unknown');
```

#### 6. Detections Table
Stores individual detection events with proper foreign key relationships.
```sql
CREATE TABLE detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    camera_id VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
);
```

#### 7. Demographics Table
Stores demographic analysis results with proper normalization.
```sql
CREATE TABLE demographics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    detection_id INTEGER,
    timestamp DATETIME NOT NULL,
    age INTEGER CHECK (age >= 0 AND age <= 120),
    age_group_id INTEGER,
    gender_id INTEGER NOT NULL,
    race_id INTEGER,
    emotion_id INTEGER,
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    analysis_model VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE SET NULL,
    FOREIGN KEY (age_group_id) REFERENCES age_groups(id),
    FOREIGN KEY (gender_id) REFERENCES genders(id),
    FOREIGN KEY (race_id) REFERENCES races(id),
    FOREIGN KEY (emotion_id) REFERENCES emotions(id)
);
```

#### 8. Dwell Times Table
Properly normalized dwell time tracking.
```sql
CREATE TABLE dwell_times (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL,
    detection_id INTEGER,
    zone_name VARCHAR(100),
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    total_time REAL, -- in seconds
    is_active BOOLEAN DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE,
    FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE SET NULL
);
```

#### 9. Analytics Sessions Table
Tracks analytics computation sessions for performance optimization.
```sql
CREATE TABLE analytics_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_start DATETIME NOT NULL,
    session_end DATETIME,
    total_detections INTEGER DEFAULT 0,
    total_persons INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active', -- active, completed, failed
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### User Management Tables (Already 3NF Compliant)

#### 10. Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name VARCHAR(100),
    role VARCHAR(20) DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until DATETIME,
    password_changed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

#### 11. User Sessions Table
```sql
CREATE TABLE user_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

#### 12. Login Attempts Table
```sql
CREATE TABLE login_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50),
    ip_address VARCHAR(45),
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    failure_reason VARCHAR(100)
);
```

## Indexes for Performance

### Primary Indexes
```sql
-- Detection performance
CREATE INDEX idx_detections_person_timestamp ON detections(person_id, timestamp);
CREATE INDEX idx_detections_timestamp ON detections(timestamp);
CREATE INDEX idx_detections_confidence ON detections(confidence);

-- Demographics performance
CREATE INDEX idx_demographics_person_timestamp ON demographics(person_id, timestamp);
CREATE INDEX idx_demographics_timestamp ON demographics(timestamp);
CREATE INDEX idx_demographics_age_group ON demographics(age_group_id);
CREATE INDEX idx_demographics_gender ON demographics(gender_id);

-- Dwell times performance
CREATE INDEX idx_dwell_times_person ON dwell_times(person_id);
CREATE INDEX idx_dwell_times_start_time ON dwell_times(start_time);
CREATE INDEX idx_dwell_times_active ON dwell_times(is_active);

-- User management performance
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_login_attempts_username ON login_attempts(username);
CREATE INDEX idx_login_attempts_ip ON login_attempts(ip_address);
```

## Data Relationships

### Entity Relationship Diagram
```
persons (1) ←→ (many) detections
persons (1) ←→ (many) demographics  
persons (1) ←→ (many) dwell_times

age_groups (1) ←→ (many) demographics
genders (1) ←→ (many) demographics
races (1) ←→ (many) demographics
emotions (1) ←→ (many) demographics

detections (1) ←→ (0..1) demographics
detections (1) ←→ (0..1) dwell_times

users (1) ←→ (many) user_sessions
```

## 3NF Compliance Verification

### 1NF Compliance ✅
- All table cells contain atomic values
- Each row is unique (enforced by PRIMARY KEY)
- No repeating groups

### 2NF Compliance ✅
- All tables have proper primary keys
- All non-key attributes are fully functionally dependent on the primary key
- No partial dependencies exist

### 3NF Compliance ✅
- No transitive dependencies
- Age groups are normalized into lookup table
- Gender, emotion, and race classifications are normalized
- All calculated fields removed from storage tables
- Proper foreign key relationships established

## Migration Strategy

### Phase 1: Create New Tables
1. Create lookup tables (age_groups, genders, emotions, races)
2. Create new normalized tables
3. Populate lookup tables with default data

### Phase 2: Data Migration
1. Migrate existing data to new schema
2. Update age_group_id based on age calculations
3. Map text values to lookup table IDs
4. Verify data integrity

### Phase 3: Update Application Code
1. Update database access layer
2. Modify queries to use joins with lookup tables
3. Update API endpoints to return normalized data
4. Test all functionality

### Phase 4: Cleanup
1. Drop old tables after verification
2. Update documentation
3. Performance testing

## Benefits of 3NF Schema

1. **Data Integrity**: Foreign key constraints ensure referential integrity
2. **Reduced Redundancy**: Lookup tables eliminate duplicate text storage
3. **Consistency**: Standardized classifications for age groups, genders, etc.
4. **Performance**: Proper indexing and normalized structure improve query performance
5. **Maintainability**: Easier to modify classifications without data migration
6. **Scalability**: Better structure for handling large datasets
7. **Reporting**: Improved analytical capabilities with proper relationships

## Query Examples

### Get Demographics with Lookup Values
```sql
SELECT 
    d.id,
    d.timestamp,
    p.id as person_id,
    d.age,
    ag.group_name as age_group,
    g.display_name as gender,
    r.display_name as race,
    e.display_name as emotion,
    d.confidence
FROM demographics d
JOIN persons p ON d.person_id = p.id
LEFT JOIN age_groups ag ON d.age_group_id = ag.id
JOIN genders g ON d.gender_id = g.id
LEFT JOIN races r ON d.race_id = r.id
LEFT JOIN emotions e ON d.emotion_id = e.id
WHERE d.timestamp >= datetime('now', '-24 hours')
ORDER BY d.timestamp DESC;
```

### Analytics Summary Query
```sql
SELECT 
    COUNT(DISTINCT d.person_id) as total_visitors,
    AVG(d.confidence) as avg_confidence,
    g.display_name as gender,
    COUNT(*) as gender_count
FROM demographics d
JOIN genders g ON d.gender_id = g.id
WHERE d.timestamp >= datetime('now', '-24 hours')
GROUP BY g.id, g.display_name;
``` 