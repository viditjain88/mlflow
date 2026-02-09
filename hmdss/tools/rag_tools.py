import json
from crewai.tools import BaseTool
from pydantic import Field, ConfigDict
from typing import Any, List, Optional

class MockVectorSearchTool(BaseTool):
    name: str = "Vector Search Tool"
    description: str = "Searches the vector database for relevant documents."

    def _run(self, query: str) -> str:
        # In a real scenario, this uses Qdrant/Chroma
        # Here we return simulated docs
        # We return a JSON string of a list of dicts
        results = [
            {"content": f"Document about {query} - Section 1", "source": "policy.pdf", "score": 0.95},
            {"content": f"Details regarding {query} - Section 2", "source": "logs.csv", "score": 0.88},
            {"content": f"Irrelevant info regarding {query}", "source": "random.txt", "score": 0.1}
        ]
        return json.dumps(results)

class BranchingRAGTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Branching RAG Tool"
    description: str = "Decomposes complex queries into sub-queries and retrieves documents for each."
    llm: Any = Field(description="The LLM to use for reasoning")
    search_tool: Any = Field(description="The underlying search tool")

    def _run(self, query: str) -> str:
        # 1. Decompose query using LLM
        prompt = f"Decompose the following healthcare query into 2-3 distinct sub-queries. Return them as a comma-separated list. Query: {query}"

        if hasattr(self.llm, 'predict'):
            sub_queries_str = self.llm.predict(prompt)
        elif hasattr(self.llm, 'invoke'):
            res = self.llm.invoke(prompt)
            sub_queries_str = res.content if hasattr(res, 'content') else str(res)
        else:
            # Mock behavior
            sub_queries_str = f"staffing for {query}, supply for {query}"

        sub_queries = [q.strip() for q in sub_queries_str.split(',')]

        all_results = []
        for sub_q in sub_queries:
            try:
                # The search_tool might return a string (JSON) or structured data
                # CrewAI tools always return strings via run, but check run vs _run usage
                res_str = self.search_tool.run(sub_q)
                res_list = json.loads(res_str)
                all_results.extend(res_list)
            except Exception as e:
                # Log error or skip
                pass

        # Deduplicate
        seen = set()
        unique_results = []
        for r in all_results:
            content = r.get('content', '')
            if content not in seen:
                seen.add(content)
                unique_results.append(r)

        return json.dumps(unique_results)

class IterativeRAGTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Iterative RAG Tool"
    description: str = "Performs multi-round retrieval to deepen context."
    llm: Any = Field(description="The LLM to use for reasoning")
    search_tool: Any = Field(description="The underlying search tool")

    def _run(self, query: str) -> str:
        # Round 1
        res_str = self.search_tool.run(query)
        initial_results = json.loads(res_str)

        # Analyze and refine
        context = " ".join([r.get('content','') for r in initial_results[:2]])
        prompt = f"Based on the initial search results: '{context}', generate a follow-up query to find missing details for: {query}"

        if hasattr(self.llm, 'predict'):
            new_query = self.llm.predict(prompt)
        elif hasattr(self.llm, 'invoke'):
            res = self.llm.invoke(prompt)
            new_query = res.content if hasattr(res, 'content') else str(res)
        else:
            new_query = f"detailed protocols for {query}"

        # Round 2
        res2_str = self.search_tool.run(new_query)
        res2 = json.loads(res2_str)

        combined = initial_results + res2

        # Deduplicate
        seen = set()
        unique_results = []
        for r in combined:
            content = r.get('content', '')
            if content not in seen:
                seen.add(content)
                unique_results.append(r)

        return json.dumps(unique_results)

class SelfReflectiveRAGTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "Self-Reflective RAG Tool"
    description: str = "Validates relevance of retrieved documents."
    llm: Any = Field(description="The LLM to use for reasoning")
    search_tool: Any = Field(description="The underlying search tool")

    def _run(self, query: str) -> str:
        res_str = self.search_tool.run(query)
        results = json.loads(res_str)

        validated_results = []
        for doc in results:
            content = doc.get('content', '')
            prompt = f"Is the following document relevant to '{query}'? Answer YES or NO.\nDocument: {content}"

            if hasattr(self.llm, 'predict'):
                ans = self.llm.predict(prompt)
            elif hasattr(self.llm, 'invoke'):
                res = self.llm.invoke(prompt)
                ans = res.content if hasattr(res, 'content') else str(res)
            else:
                # Mock logic
                if "Irrelevant" in content:
                    ans = "NO"
                else:
                    ans = "YES"

            if "YES" in ans.upper():
                validated_results.append(doc)

        return json.dumps(validated_results)
