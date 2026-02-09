import os
import argparse
from crewai import LLM
from hmdss.scripts.generate_data import ensure_dirs, generate_pdf_policy, generate_csv_logs, generate_json_ehr
from hmdss.analytics.train import train_model
from hmdss.core.workflow import HMDSSWorkflow

def setup_environment():
    print("Checking environment...")
    if not os.path.exists("hmdss/data/policies/surgical_policy.pdf"):
        print("Generating data...")
        ensure_dirs()
        generate_pdf_policy()
        generate_csv_logs()
        generate_json_ehr()

    if not os.path.exists("hmdss/analytics/patient_volume_model.pth"):
        print("Training ANN model...")
        train_model()

def main():
    parser = argparse.ArgumentParser(description="Healthcare Management Decision Support System (HMDSS)")
    parser.add_argument("--query", type=str, default="Optimize surgical scheduling for next Monday", help="The query to process")
    parser.add_argument("--model", type=str, default="ollama/gemma:2b", help="LLM model to use (e.g., ollama/gemma:2b, gpt-4)")
    parser.add_argument("--api_base", type=str, default="http://localhost:11434", help="API base URL for local LLM")

    args = parser.parse_args()

    setup_environment()

    print(f"Initializing Workflow with model: {args.model}")

    # Configure LLM
    # Note: For Ollama, we set base_url. For OpenAI, we need API key in env.
    llm = LLM(
        model=args.model,
        base_url=args.api_base,
        api_key="NA" # Dummy if using local
    )

    workflow = HMDSSWorkflow(llm)

    print(f"Processing Query: {args.query}")
    try:
        result = workflow.run(args.query)
        print("\n--- Final Recommendation ---\n")
        print(result)
    except Exception as e:
        print(f"Error during execution: {e}")
        # Check for common connection errors
        if "Connection refused" in str(e):
             print("\nTip: Ensure your local LLM (Ollama) is running.")

if __name__ == "__main__":
    main()
