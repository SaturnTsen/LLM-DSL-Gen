import os
from dsl_gen.core import build_rag_flow, CFG, RAGState_to_dict
import os
from pathlib import Path
import json
from datetime import datetime

flow = build_rag_flow()

challenges_path = CFG.PATH_CFG.CHALLENGES_PATH

# Load the challenges

results = {
    "CONFIG": CFG.to_dict(),
    "total challenges": 0,
    "success": 0,
    "pending": 0,  # This indicate that the challenge is not finished, check if there is a bug
    "compilation error": 0,
    "judgment error": 0,
    "failed challenges": {
        "compilation error": [],
        "judgment error": []
    },
    "details": [
    ]
}
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Prompt the user to continue
user_input = input("Ensure that the vector db corresponds to the latest retrieval method. Do you need me cleaning cached vectordb ? (y/n): ")
if user_input.lower() == 'y':
    # Clear the files in the directory
    vector_db_dir = Path(CFG.VECTOR_DB_DIR)
    for file in vector_db_dir.iterdir():
        if file.is_file():
            file.unlink()
        elif file.is_dir():
            for f in file.iterdir():
                if f.is_file():
                    f.unlink()
            file.rmdir()

# Run the benchmarks
for file_name in os.listdir(challenges_path):
    if not file_name.endswith(".json"):
        continue
    if not file_name.endswith("14.json") and not file_name.endswith("15.json"):
        continue
    print(f"[PIPELINE] Processing {file_name}")
    selected_file = Path(challenges_path) / file_name
    result = flow.invoke({"challenge_path": selected_file})

    results["total challenges"] += 1
    if result["final_state"] == "success":
        results["success"] += 1
    elif result["final_state"] == "compilation error":
        results["compilation error"] += 1
        results["failed challenges"]["compilation error"].append({
            "challenge": file_name,
            "completion": result["completion"],
            "error": result["compilation_result"]["messages"],
            "ground_truth": result["ground_truth"]
        })
    elif result["final_state"] == "judgment error":
        results["judgment error"] += 1
        results["failed challenges"]["judgment error"].append({
            "challenge": file_name,
            "completion": result["completion"],
            "error": result["judge_output"],
            "ground_truth": result["ground_truth"]
        })
    elif result["final_state"] == "pending":
        results["pending"] += 1

    print(result)
    results["details"].append(RAGState_to_dict(result))
    # Get the current system time

    # Create logs directory if it doesn't exist
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Save the results to a JSON file in the logs directory
    log_file = logs_dir / f"results_{current_time}.json"

    print(f"Stashing results in {log_file}")
    with open(log_file, "w") as f:
        json.dump(results, f, indent=4)
