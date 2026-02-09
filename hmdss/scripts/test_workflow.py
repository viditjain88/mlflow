from unittest.mock import MagicMock, patch
from crewai import LLM
from hmdss.core.workflow import HMDSSWorkflow
import json
import os

os.environ["OPENAI_API_KEY"] = "sk-proj-dummy"

def test_workflow_mocked():
    # Instantiate LLM
    llm = LLM(model="gpt-3.5-turbo", api_key="dummy")

    # Mock the call method on this instance
    llm.call = MagicMock(return_value="Final Recommendation based on fusion.")

    # Also mock 'predict' or 'invoke' if used?
    # HMDSSWorkflow checks hasattr(llm, 'predict') etc.
    # crewai.LLM usually has 'call'.
    # But let's verify what methods it has.
    # We can just add them to the mock if needed.
    llm.predict = MagicMock(return_value="Final Recommendation based on fusion.")
    llm.invoke = MagicMock(return_value=MagicMock(content="Final Recommendation based on fusion."))

    with patch('crewai.Crew.kickoff') as mock_kickoff:
        mock_docs = [
            {"content": "Doc 1", "source": "s1", "score": 0.9},
            {"content": "Doc 2", "source": "s2", "score": 0.8}
        ]

        mock_result = MagicMock()
        mock_result.raw = json.dumps(mock_docs)
        mock_kickoff.return_value = mock_result

        workflow = HMDSSWorkflow(llm)

        query = "Optimize surgical staffing"
        result = workflow.run(query)

        print("Final Result:", result)
        assert "Final Recommendation" in result

        # Verify call was made
        # workflow.run calls llm.predict/invoke/call
        # We mocked all 3.

        # Check which one was called
        if llm.predict.called:
            args = llm.predict.call_args[0]
            print("Called predict")
        elif llm.invoke.called:
            args = llm.invoke.call_args[0]
            print("Called invoke")
        elif llm.call.called:
            args = llm.call.call_args[0]
            print("Called call")
        else:
            print("LLM not called?")

        prompt = str(args[0])
        assert "Doc 1" in prompt
        assert "Doc 2" in prompt

if __name__ == "__main__":
    test_workflow_mocked()
