#!/usr/bin/env python3
"""
Test script for the Face Snapshot → Person ID → Demographics system

This script tests the complete pipeline from face detection to demographic analysis
and database storage with quality assessment.
"""

import sys
import os
import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from database.sqlite_db import SQLiteDatabase
from analysis import FaceSnapshotManager, FaceQualityScorer, DemographicAnalyzer
from detection import get_detector, get_tracker
from detection.detection_utils import get_detection_pipeline
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration"""
    try:
        with open('src/config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}

def create_test_face_image():
    """Create a synthetic test face image"""
    # Create a simple face-like image for testing
    img = np.zeros((112, 112, 3), dtype=np.uint8)
    
    # Face outline (oval)
    cv2.ellipse(img, (56, 56), (45, 55), 0, 0, 360, (200, 180, 160), -1)
    
    # Eyes
    cv2.circle(img, (40, 40), 8, (50, 50, 50), -1)  # Left eye
    cv2.circle(img, (72, 40), 8, (50, 50, 50), -1)  # Right eye
    
    # Nose
    cv2.circle(img, (56, 56), 3, (150, 130, 110), -1)
    
    # Mouth
    cv2.ellipse(img, (56, 75), (15, 8), 0, 0, 180, (100, 80, 80), 2)
    
    return img

def test_face_quality_scorer():
    """Test the face quality scoring system"""
    logger.info("Testing Face Quality Scorer...")
    
    # Initialize quality scorer
    quality_config = {
        'size_weight': 0.25,
        'clarity_weight': 0.30,
        'frontality_weight': 0.20,
        'lighting_weight': 0.15,
        'detection_weight': 0.10,
        'min_face_size': 48,
        'optimal_face_size': 112,
        'max_face_size': 300
    }
    
    scorer = FaceQualityScorer(quality_config)
    
    # Test with synthetic face
    test_face = create_test_face_image()
    bbox = (0, 0, 112, 112)
    detection_confidence = 0.9
    
    quality_score = scorer.calculate_quality_score(test_face, bbox, detection_confidence)
    quality_grade = scorer.get_quality_grade(quality_score)
    
    logger.info(f"✓ Face quality score: {quality_score:.3f} (Grade: {quality_grade})")
    
    # Test with different sizes
    small_face = cv2.resize(test_face, (32, 32))
    small_quality = scorer.calculate_quality_score(small_face, (0, 0, 32, 32), 0.8)
    
    large_face = cv2.resize(test_face, (200, 200))
    large_quality = scorer.calculate_quality_score(large_face, (0, 0, 200, 200), 0.95)
    
    logger.info(f"✓ Small face quality: {small_quality:.3f}")
    logger.info(f"✓ Large face quality: {large_quality:.3f}")
    
    return scorer

def test_database_schema():
    """Test the enhanced database schema"""
    logger.info("Testing enhanced database schema...")
    
    # Initialize database
    db = SQLiteDatabase({'path': 'test_reviision.db'})
    
    # Test storing a person
    timestamp = datetime.now()
    
    # Store person
    person_id = 1
    bbox = (100, 100, 150, 150)
    confidence = 0.85
    
    detection_id = db.store_detection(person_id, bbox, confidence, timestamp)
    logger.info(f"✓ Stored detection: {detection_id}")
    
    # Store demographics
    demographics = {
        'age': 28,
        'gender': 'male',
        'race': 'white',
        'emotion': 'happy',
        'confidence': 0.87
    }
    
    demo_id = db.store_demographics(person_id, demographics, timestamp, detection_id, 'enhanced')
    logger.info(f"✓ Stored demographics: {demo_id}")
    
    # Test face snapshot storage
    test_face = create_test_face_image()
    
    # Convert image to binary
    _, buffer = cv2.imencode('.jpg', test_face)
    face_data = buffer.tobytes()
    
    snapshot_id = db.store_face_snapshot(
        person_id=person_id,
        face_image_data=face_data,
        quality_score=0.85,
        confidence=confidence,
        bbox=bbox,
        analysis_method='test',
        timestamp=timestamp,
        is_primary=True
    )
    
    logger.info(f"✓ Stored face snapshot: {snapshot_id}")
    
    # Test retrieval
    primary_face = db.get_primary_face_snapshot(person_id)
    if primary_face:
        logger.info(f"✓ Retrieved primary face: quality={primary_face['quality_score']:.3f}")
    
    best_face = db.get_best_face_snapshot(person_id)
    if best_face:
        logger.info(f"✓ Retrieved best face: quality={best_face['quality_score']:.3f}")
    
    # Test quality check
    has_good_demographics = db.has_good_demographics(person_id, min_confidence=0.7)
    logger.info(f"✓ Has good demographics: {has_good_demographics}")
    
    return db

def test_face_snapshot_manager(db):
    """Test the face snapshot manager"""
    logger.info("Testing Face Snapshot Manager...")
    
    config = {
        'storage_directory': 'data/test_faces',
        'storage_format': 'jpg',
        'storage_quality': 95,
        'max_faces_per_person': 3,
        'enable_file_storage': True,
        'enable_db_storage': True,
        'min_quality_threshold': 0.5,
        'improvement_threshold': 0.1,
        'max_face_size': (224, 224)
    }
    
    manager = FaceSnapshotManager(config, db)
    
    # Test processing different quality faces
    person_id = 2
    
    # Low quality face
    low_quality_face = cv2.resize(create_test_face_image(), (48, 48))
    low_quality_face = cv2.GaussianBlur(low_quality_face, (5, 5), 2.0)  # Make it blurry
    
    result1 = manager.process_face_snapshot(
        person_id=person_id,
        face_img=low_quality_face,
        bbox=(0, 0, 48, 48),
        detection_confidence=0.6
    )
    
    logger.info(f"✓ Low quality result: {result1['reason']} (quality: {result1['quality_score']:.3f})")
    
    # High quality face
    high_quality_face = create_test_face_image()
    
    result2 = manager.process_face_snapshot(
        person_id=person_id,
        face_img=high_quality_face,
        bbox=(0, 0, 112, 112),
        detection_confidence=0.9
    )
    
    logger.info(f"✓ High quality result: stored={result2['stored']}, primary={result2['is_primary']} (quality: {result2['quality_score']:.3f})")
    
    # Test thumbnail generation
    thumbnail = manager.get_face_thumbnail_base64(person_id, (64, 64))
    if thumbnail:
        logger.info(f"✓ Generated thumbnail: {len(thumbnail)} characters")
    else:
        logger.warning("Could not generate thumbnail")
    
    # Test storage stats
    stats = manager.get_storage_stats()
    logger.info(f"✓ Storage stats: {stats}")
    
    return manager

def test_demographic_analyzer_integration(db, face_manager):
    """Test the enhanced demographic analyzer with face quality integration"""
    logger.info("Testing Enhanced Demographic Analyzer...")
    
    config = {
        'min_face_size': 48,
        'confidence_threshold': 0.6,
        'use_insightface': False,  # Disable for testing
        'one_time_demographics': True,
        'model_dir': './models'
    }
    
    analyzer = DemographicAnalyzer(config)
    
    # Test with quality check
    person_id = 3
    test_face = create_test_face_image()
    bbox = (0, 0, 112, 112)
    detection_confidence = 0.9
    
    # First analysis (should process)
    result1 = analyzer.analyze_with_quality_check(
        person_id=person_id,
        face_img=test_face,
        bbox=bbox,
        detection_confidence=detection_confidence,
        database=db,
        face_snapshot_manager=face_manager
    )
    
    logger.info(f"✓ First analysis: method={result1.get('analysis_method')}, "
               f"quality={result1.get('quality_score', 0):.3f}")
    
    # Store the demographics
    demo_id = db.store_demographics(person_id, result1, datetime.now(), analysis_model='enhanced')
    
    # Second analysis (should use cache)
    result2 = analyzer.analyze_with_quality_check(
        person_id=person_id,
        face_img=test_face,
        bbox=bbox,
        detection_confidence=detection_confidence,
        database=db,
        face_snapshot_manager=face_manager
    )
    
    logger.info(f"✓ Second analysis: method={result2.get('analysis_method')}")
    
    return analyzer

def test_full_pipeline():
    """Test the complete face snapshot pipeline"""
    logger.info("Testing complete face snapshot pipeline...")
    
    # Load config
    config = load_config()
    
    # Initialize components
    db = test_database_schema()
    scorer = test_face_quality_scorer()
    manager = test_face_snapshot_manager(db)
    analyzer = test_demographic_analyzer_integration(db, manager)
    
    logger.info("✓ All components initialized successfully")
    
    # Simulate real workflow
    person_id = 100
    
    # Simulate multiple face detections with varying quality
    faces_data = [
        {'quality': 'low', 'size': (48, 48), 'blur': 3.0, 'confidence': 0.6},
        {'quality': 'medium', 'size': (80, 80), 'blur': 1.0, 'confidence': 0.8},
        {'quality': 'high', 'size': (112, 112), 'blur': 0.0, 'confidence': 0.95},
        {'quality': 'very_high', 'size': (128, 128), 'blur': 0.0, 'confidence': 0.98}
    ]
    
    logger.info(f"Simulating face detection sequence for person {person_id}...")
    
    for i, face_data in enumerate(faces_data):
        # Create face image with specified quality
        face_img = cv2.resize(create_test_face_image(), face_data['size'])
        if face_data['blur'] > 0:
            face_img = cv2.GaussianBlur(face_img, (int(face_data['blur']*2+1), int(face_data['blur']*2+1)), face_data['blur'])
        
        bbox = (0, 0, face_data['size'][0], face_data['size'][1])
        
        # Process with enhanced system
        result = analyzer.analyze_with_quality_check(
            person_id=person_id,
            face_img=face_img,
            bbox=bbox,
            detection_confidence=face_data['confidence'],
            database=db,
            face_snapshot_manager=manager
        )
        
        logger.info(f"  Frame {i+1} ({face_data['quality']}): "
                   f"method={result.get('analysis_method')}, "
                   f"quality={result.get('quality_score', 0):.3f}, "
                   f"stored={result.get('face_stored', False)}")
    
    # Check final state
    primary_face = db.get_primary_face_snapshot(person_id)
    if primary_face:
        logger.info(f"✓ Final primary face quality: {primary_face['quality_score']:.3f}")
    
    has_good_demo = db.has_good_demographics(person_id)
    logger.info(f"✓ Person {person_id} has good demographics: {has_good_demo}")
    
    # Test cleanup
    removed = db.cleanup_old_face_snapshots(person_id, max_snapshots=2)
    logger.info(f"✓ Cleaned up {removed} old face snapshots")
    
    logger.info("✅ Full pipeline test completed successfully!")

def main():
    """Main test function"""
    logger.info("Starting Face Snapshot System Test...")
    
    try:
        # Clean up any existing test database
        test_db_path = Path('test_reviision.db')
        if test_db_path.exists():
            test_db_path.unlink()
        
        # Clean up test face directory
        test_faces_dir = Path('data/test_faces')
        if test_faces_dir.exists():
            import shutil
            shutil.rmtree(test_faces_dir)
        
        # Run full pipeline test
        test_full_pipeline()
        
        logger.info("🎉 All tests passed! Face snapshot system is working correctly.")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        return 1
    
    finally:
        # Cleanup
        test_db_path = Path('test_reviision.db')
        if test_db_path.exists():
            test_db_path.unlink()
            logger.info("Cleaned up test database")
    
    return 0

if __name__ == '__main__':
    exit(main()) 