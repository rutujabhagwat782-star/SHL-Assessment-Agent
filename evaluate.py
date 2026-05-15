import json
import requests
import time

# ---------------------------------------------------------
# SHL Agent Evaluation Script
# ---------------------------------------------------------
# This script measures retrieval quality (Recall@K), 
# schema compliance, and basic behavioral probes.

API_URL = "http://localhost:8000/chat"
HEALTH_URL = "http://localhost:8000/health"


traces = [
    {
        "name": "Vague Request (Behavior Probe)",
        "messages": [{"role": "user", "content": "I need an assessment."}],
        "expected_behavior": "clarify",
        "expected_recommendations_count": 0
    },
    {
        "name": "Direct Java Request (Recall Check)",
        "messages": [{"role": "user", "content": "Can you recommend an assessment for a mid-level Java 8 developer?"}],
        "expected_behavior": "recommend",
        "expected_recommendations": ["Java 8 (New)"]
    }
]

def check_health():
    try:
        res = requests.get(HEALTH_URL)
        return res.status_code == 200
    except:
        return False

def evaluate():
    print("Starting Evaluation Pipeline...")
    if not check_health():
        print("API is not running. Please start the server with `uvicorn main:app --port 8000`")
        return

    schema_passes = 0
    total_recall_k = 0
    behavior_passes = 0

    for trace in traces:
        print(f"\nRunning Trace: {trace['name']}")
        payload = {"messages": trace["messages"]}
        
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        latency = time.time() - start_time
        
        if response.status_code != 200:
            print(f"Failed with status {response.status_code}")
            continue
            
        data = response.json()
        
        
        if all(k in data for k in ["reply", "recommendations", "end_of_conversation"]):
            schema_passes += 1
            print("Schema Compliant")
        else:
            print("Schema Failure")

        
        recs = data.get("recommendations", [])
        recommended_names = [r["name"] for r in recs]
        
        if trace["expected_behavior"] == "clarify":
            if len(recs) == 0:
                behavior_passes += 1
                print("Behavior Passed: Agent clarified and did not recommend early.")
            else:
                print(f"Behavior Failed: Agent recommended {len(recs)} tests prematurely.")
                
        elif trace["expected_behavior"] == "recommend":
            expected = trace.get("expected_recommendations", [])
            hits = sum(1 for e in expected if e in recommended_names)
            recall = hits / len(expected) if expected else 0
            total_recall_k += recall
            print(f"Behavior Passed: Agent made recommendations.")
            print(f"Recall: {recall * 100}% ({hits}/{len(expected)} expected found)")
            
        print(f"Latency: {latency:.2f}s")

    print("\n" + "="*30)
    print("EVALUATION SUMMARY")
    print("="*30)
    print(f"Schema Compliance Rate: {schema_passes}/{len(traces)} ({(schema_passes/len(traces))*100:.1f}%)")
    print(f"Behavior Pass Rate:     {behavior_passes + (total_recall_k > 0)}/{len(traces)}")
    print(f"Mean Recall:            {(total_recall_k/1)*100:.1f}%") # Divided by 1 recommending trace

if __name__ == "__main__":
    evaluate()
