import unittest
import os
import json
import torch
import shutil
from unittest.mock import MagicMock, patch

# Import HMDSS modules
from hmdss.scripts.generate_data import generate_pdf_policy, generate_csv_logs, generate_json_ehr, ensure_dirs
from hmdss.analytics.train import train_model
from hmdss.tools.ann_tool import ANNPredictionTool
from hmdss.tools.rag_tools import BranchingRAGTool, IterativeRAGTool, SelfReflectiveRAGTool, MockVectorSearchTool
from hmdss.core.workflow import HMDSSWorkflow
from hmdss.core.fusion import reciprocal_rank_fusion

class TestHMDSSPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a temp directory for data to avoid cluttering real data
        cls.test_data_dir = "hmdss/test_data"
        if not os.path.exists(cls.test_data_dir):
            os.makedirs(cls.test_data_dir)
        ensure_dirs()

    def test_01_data_generation(self):
        print("\nTesting Data Generation...")
        generate_pdf_policy()
        generate_csv_logs()
        generate_json_ehr()

        self.assertTrue(os.path.exists("hmdss/data/policies/surgical_policy.pdf"))
        self.assertTrue(os.path.exists("hmdss/data/logs/resource_logs.csv"))
        self.assertTrue(os.path.exists("hmdss/data/ehr/metadata.json"))

    def test_02_ann_training_and_prediction(self):
        print("\nTesting ANN Training...")
        # Train model
        train_model()
        self.assertTrue(os.path.exists("hmdss/analytics/patient_volume_model.pth"))

        print("Testing ANN Tool...")
        tool = ANNPredictionTool()
        prediction = tool._run("2023-12-25")
        self.assertIn("Predicted Patient Volume", prediction)

    def test_03_rag_tools_logic(self):
        print("\nTesting RAG Strategies...")

        class MockLLM:
            def predict(self, prompt):
                if "Decompose" in prompt:
                    return "staffing, supply"
                if "follow-up query" in prompt:
                    return "protocols"
                if "relevant" in prompt:
                    return "YES" if "Irrelevant" not in prompt else "NO"
                return "Unknown"
            def invoke(self, prompt):
                class Res:
                    content = "staffing, supply"
                return Res()

        llm = MockLLM()
        search_tool = MockVectorSearchTool()

        branching = BranchingRAGTool(llm=llm, search_tool=search_tool)
        res_b = branching.run("surgery")
        self.assertIn("staffing", res_b)

        iterative = IterativeRAGTool(llm=llm, search_tool=search_tool)
        res_i = iterative.run("surgery")
        self.assertIn("Document about", res_i)

        reflective = SelfReflectiveRAGTool(llm=llm, search_tool=search_tool)
        res_r = reflective.run("surgery")
        self.assertNotIn("Irrelevant", res_r)

    def test_04_fusion_logic(self):
        print("\nTesting Fusion...")
        # Avoid ties
        # List1: A(0), B(1) -> A: 1/61, B: 1/62
        list1 = [{"content": "A", "score": 10}, {"content": "B", "score": 9}]
        # List2: B(0), C(1), A(2) -> B: 1/61, C: 1/62, A: 1/63
        list2 = [{"content": "B", "score": 100}, {"content": "C", "score": 50}, {"content": "A", "score": 5}]

        # A Score: 1/61 + 1/63 = 0.01639 + 0.01587 = 0.03226
        # B Score: 1/62 + 1/61 = 0.01612 + 0.01639 = 0.03251
        # B > A

        fused = reciprocal_rank_fusion([list1, list2])
        self.assertEqual(fused[0]['content'], "B")
        self.assertEqual(fused[1]['content'], "A")

    @patch('crewai.agent.core.create_llm', side_effect=lambda x: x)
    @patch('crewai.Crew.kickoff')
    def test_05_workflow_orchestration(self, mock_kickoff, mock_create_llm):
        print("\nTesting Workflow Orchestration...")

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.predict.return_value = "Final Recommendation"
        mock_llm.call.return_value = "Final Recommendation" # For workflow usage

        # We need mock_llm to have attributes required by Agent if Agent uses them
        # But Agent mainly uses it via LLM class methods.
        # Since we patch create_llm, the agent.llm will be this mock_llm.

        # Mock Crew result
        mock_result = MagicMock()
        mock_result.raw = json.dumps([{"content": "Doc 1", "source": "s1"}])
        mock_kickoff.return_value = mock_result

        workflow = HMDSSWorkflow(mock_llm)
        result = workflow.run("test query")

        self.assertIn("Final Recommendation", result)

if __name__ == '__main__':
    unittest.main()
