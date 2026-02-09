from hmdss.tools.rag_tools import MockVectorSearchTool, BranchingRAGTool, IterativeRAGTool, SelfReflectiveRAGTool
import json

class MockLLM:
    def predict(self, prompt):
        if "Decompose" in prompt:
            return "staffing, supply"
        if "follow-up query" in prompt:
            return "detailed protocols"
        if "relevant" in prompt:
            if "Irrelevant" in prompt:
                return "NO"
            return "YES"
        return "Unknown"

def test_rag_tools():
    llm = MockLLM()
    search_tool = MockVectorSearchTool()

    # Test Branching
    branching_tool = BranchingRAGTool(llm=llm, search_tool=search_tool)
    res_b = branching_tool.run("surgery")
    print("Branching Result:", res_b)
    assert "staffing" in res_b or "supply" in res_b or "Document about" in res_b

    # Test Iterative
    iter_tool = IterativeRAGTool(llm=llm, search_tool=search_tool)
    res_i = iter_tool.run("surgery")
    print("Iterative Result:", res_i)

    # Test Self-Reflective
    reflect_tool = SelfReflectiveRAGTool(llm=llm, search_tool=search_tool)
    res_r = reflect_tool.run("surgery")
    print("Reflective Result:", res_r)
    # Check that irrelevant info is removed
    assert "Irrelevant" not in res_r

if __name__ == "__main__":
    test_rag_tools()
