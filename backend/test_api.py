import requests
import json
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:5000/api"

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*50)
    print("Testing Health Check...")
    print("="*50)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_model_info():
    """Test the model info endpoint"""
    print("\n" + "="*50)
    print("Testing Model Info...")
    print("="*50)
    
    response = requests.get(f"{API_BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_config():
    """Test the config endpoint"""
    print("\n" + "="*50)
    print("Testing Configuration...")
    print("="*50)
    
    response = requests.get(f"{API_BASE_URL}/config")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_damage_detection(pre_image_path, post_image_path, threshold=0.3):
    """Test the damage detection endpoint"""
    print("\n" + "="*50)
    print("Testing Damage Detection...")
    print("="*50)
    
    # Check if files exist
    if not Path(pre_image_path).exists():
        print(f"❌ Pre-disaster image not found: {pre_image_path}")
        return False
    
    if not Path(post_image_path).exists():
        print(f"❌ Post-disaster image not found: {post_image_path}")
        return False
    
    # Prepare files
    with open(pre_image_path, 'rb') as pre_f, open(post_image_path, 'rb') as post_f:
        files = {
            'pre_image': ('pre.png', pre_f, 'image/png'),
            'post_image': ('post.png', post_f, 'image/png')
        }
        
        data = {
            'threshold': str(threshold)
        }
        
        print(f"Uploading images...")
        print(f"  Pre-image: {pre_image_path}")
        print(f"  Post-image: {post_image_path}")
        print(f"  Threshold: {threshold}")
        
        response = requests.post(
            f"{API_BASE_URL}/detect-damage",
            files=files,
            data=data
        )
    
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n✓ Detection successful!")
        print(f"\nAnalysis ID: {result['analysis_id']}")
        print(f"Timestamp: {result['timestamp']}")
        
        stats = result['statistics']
        print(f"\n--- Statistics ---")
        print(f"Buildings Detected: {stats['buildings_detected']}")
        print(f"Total Building Pixels: {stats['total_building_pixels']:,}")
        
        print(f"\n--- Damage Breakdown ---")
        for damage_type, count in stats['damage_breakdown'].items():
            percentage = stats['damage_percentages'][damage_type]
            print(f"{damage_type}: {count:,} pixels ({percentage}%)")
        
        print(f"\n--- Summary ---")
        print(f"Most Severe Damage: {result['summary']['most_severe_damage']}")
        
        # Save visualizations
        print(f"\n--- Saving Visualizations ---")
        analysis_id = result['analysis_id']
        
        # You can decode and save base64 images if needed
        print(f"Heatmap visualization available")
        print(f"Overlay visualization available")
        print(f"Localization map available")
        
        return True
    else:
        print(f"❌ Error: {response.json()}")
        return False

def test_history():
    """Test the history endpoint"""
    print("\n" + "="*50)
    print("Testing History...")
    print("="*50)
    
    response = requests.get(f"{API_BASE_URL}/history")
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total analyses: {result['count']}")
        
        if result['count'] > 0:
            print("\nRecent analyses:")
            for analysis in result['results'][:3]:  # Show first 3
                print(f"  - {analysis['id']} ({analysis['timestamp']})")
        
        return True
    else:
        print(f"❌ Error: {response.json()}")
        return False

def run_all_tests(pre_image_path=None, post_image_path=None):
    """Run all API tests"""
    print("\n" + "#"*50)
    print("# Infrastructure Damage Detection API Tests")
    print("#"*50)
    
    results = {}
    
    # Basic endpoint tests
    results['health'] = test_health_check()
    results['model_info'] = test_model_info()
    results['config'] = test_config()
    
    # Detection test (if images provided)
    if pre_image_path and post_image_path:
        results['damage_detection'] = test_damage_detection(pre_image_path, post_image_path)
        results['history'] = test_history()
    else:
        print("\n⚠️  Skipping damage detection test (no images provided)")
        print("To test damage detection, provide image paths:")
        print("  python test_api.py path/to/pre.png path/to/post.png")
    
    # Summary
    print("\n" + "#"*50)
    print("# Test Summary")
    print("#"*50)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(results.values())

if __name__ == "__main__":
    import sys
    
    # Get image paths from command line if provided
    pre_image = '/media/piyush/DATA/Change_Detection/geosense/sadf/Pasted image.png'
    post_image = '/media/piyush/DATA/Change_Detection/geosense/sadf/Pasted image (2).png'
    
    success = run_all_tests(pre_image, post_image)
    sys.exit(0 if success else 1)
