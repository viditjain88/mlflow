from crewai import Agent
from hmdss.tools.rag_tools import BranchingRAGTool, IterativeRAGTool, SelfReflectiveRAGTool
from hmdss.tools.ann_tool import ANNPredictionTool

class HMDSSAgents:
    def __init__(self, llm, search_tool):
        self.llm = llm
        self.search_tool = search_tool

        # Tools
        # Note: We pass the search_tool (e.g. wrapper around VectorDB) to the specialized RAG tools
        self.branching_tool = BranchingRAGTool(llm=llm, search_tool=search_tool)
        self.iterative_tool = IterativeRAGTool(llm=llm, search_tool=search_tool)
        self.reflective_tool = SelfReflectiveRAGTool(llm=llm, search_tool=search_tool)
        self.ann_tool = ANNPredictionTool()

    def analyst(self):
        return Agent(
            role="The Analyst",
            goal="Decompose complex queries and retrieve comprehensive information using Branching RAG strategy.",
            backstory="You are an expert analyst who breaks down problems into smaller components to find the root cause.",
            tools=[self.branching_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def scholar(self):
        return Agent(
            role="The Scholar",
            goal="Perform deep research through iterative searching using Iterative RAG strategy.",
            backstory="You are a meticulous researcher who digs deep, following leads to uncover hidden details.",
            tools=[self.iterative_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def fact_checker(self):
        return Agent(
            role="The Fact-Checker",
            goal="Validate information and filter out irrelevance using Self-Reflective RAG strategy.",
            backstory="You are a strict validator ensuring only high-quality, relevant data is used.",
            tools=[self.reflective_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

    def investigator(self):
        return Agent(
            role="The Investigator",
            goal="Use predictive models and search to answer questions requiring reasoning.",
            backstory="You are a data-driven investigator who combines historical data with predictive analytics.",
            tools=[self.ann_tool, self.search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
