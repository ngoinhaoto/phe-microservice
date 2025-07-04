import requests
import json

MICROSERVICE_URL = "http://localhost:8001/test-key-compatibility"  

def main():
    print("Testing PHE key compatibility between microservice and server...")
    try:
        resp = requests.post(MICROSERVICE_URL)
        resp.raise_for_status()
        result = resp.json()
        print(json.dumps(result, indent=2))
        if result.get("key_compatibility") and result.get("key_ids_match"):
            print("\n✅ Keys are compatible and decryption is correct!")
        else:
            print("\n❌ Key mismatch or decryption error!")
            if result.get("errors"):
                print("Errors:", result["errors"])
    except Exception as e:
        print("Test failed:", e)

if __name__ == "__main__":
    main()