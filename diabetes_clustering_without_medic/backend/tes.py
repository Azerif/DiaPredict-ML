import requests
import json

# Test cases dengan berbagai format
test_cases = [{
    "name": "English Short Format",
    "data": {
            "gender": "Male",
            "age": 45,
            "hypertension": 0,
            "heart_disease": 0,
            "smoking_history": "never",  # Format pendek yang sekarang didukung
            "bmi": 25.5
    }
},    {
    "name": "Indonesian Format",
    "data": {
            "gender": "Female",
            "age": 80,
            "hypertension": 0,
            "heart_disease": 1,
            "smoking_history": "never_smoke",
            "bmi": 25.19
    }
},    {
    "name": "Full English Format",
    "data": {
            "gender": "Female",
            "age": 30,
            "hypertension": 0,
            "heart_disease": 0,
            "smoking_history": "former_smoke",
            "bmi": 22.0
    }
}
]

# URLs to try
urls_to_try = [
    'http://localhost:7860/predict',
]

print("🧪 Testing Multiple Formats...\n")

for test_case in test_cases:
    print(f"📋 Test: {test_case['name']}")
    print(f"Data: {json.dumps(test_case['data'], indent=2)}")

    success = False
    for url in urls_to_try:
        try:
            print(f"\n🔗 Trying {url}...")
            response = requests.post(url, json=test_case['data'], timeout=10)

            if response.status_code == 200:
                print("✅ SUCCESS!")
                result = response.json()
                print(
                    f"Cluster: {result['predicted_cluster']} - {result['cluster_name']}")
                print(f"Confidence: {result['confidence']:.3f}")
                success = True
                break
            else:
                print(f"❌ Error {response.status_code}")
                print(f"Response: {response.json()}")

        except requests.exceptions.ConnectionError:
            print(f"❌ Connection failed to {url}")
        except Exception as e:
            print(f"❌ Error: {e}")

    if not success:
        print("❌ All URLs failed for this test case")

    print("-" * 60)

print("\n🔍 If localhost fails, make sure server is running:")
print("uvicorn main:app --host 0.0.0.0 --port 7860 --reload")
