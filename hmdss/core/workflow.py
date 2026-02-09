from crewai import Crew, Task, Process
import concurrent.futures
import json
import re
from hmdss.agents.definitions import HMDSSAgents
from hmdss.core.fusion import reciprocal_rank_fusion
from hmdss.tools.rag_tools import MockVectorSearchTool

class HMDSSWorkflow:
    def __init__(self, llm):
        # Initialize shared search tool (mock or real)
        self.search_tool = MockVectorSearchTool()
        self.agents_factory = HMDSSAgents(llm, self.search_tool)
        self.llm = llm

    def run(self, query: str):
        # 1. Create Agents
        analyst = self.agents_factory.analyst()
        scholar = self.agents_factory.scholar()
        fact_checker = self.agents_factory.fact_checker()
        investigator = self.agents_factory.investigator()

        agents_list = [analyst, scholar, fact_checker, investigator]

        # 2. Define Tasks for each agent
        # The goal is to retrieve documents. We instruct them to return JSON.
        tasks = []
        for agent in agents_list:
            task_desc = f"Retrieve relevant documents for the query: '{query}' using your specialized tool. Return the raw JSON list of documents as your final answer."
            task = Task(
                description=task_desc,
                agent=agent,
                expected_output="A JSON string representing a list of documents."
            )
            tasks.append(task)

        # 3. Execute in Parallel
        results = []
        # CrewAI might have issues with threading if sqlite is shared.
        # But here each crew is isolated, assuming no shared state in CrewAI internals that breaks.
        # If threading fails, we fall back to sequential.

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_agent = {
                    executor.submit(self._run_single_crew, agent, task): agent.role
                    for agent, task in zip(agents_list, tasks)
                }

                for future in concurrent.futures.as_completed(future_to_agent):
                    role = future_to_agent[future]
                    try:
                        output = future.result()
                        parsed = self._extract_json(output)
                        results.append(parsed)
                    except Exception as e:
                        print(f"Agent {role} failed: {e}")
                        results.append([])
        except ImportError:
            # Fallback for environments without threading support? No, standard python has it.
            pass

        # 4. Fusion
        clean_results = [r for r in results if isinstance(r, list)]
        fused_docs = reciprocal_rank_fusion(clean_results)

        # 5. Final Synthesis
        final_context = "\n\n".join([
            f"Source: {d.get('source', 'unknown')}\nContent: {d.get('content', '')}"
            for d in fused_docs[:5] # Top 5
        ])

        final_prompt = f"""
        You are an expert Healthcare Management Decision Support System.
        Based on the following retrieved context and analysis, provide a comprehensive answer to the query.

        Query: {query}

        Context:
        {final_context}

        Recommendation:
        """

        if hasattr(self.llm, 'predict'):
            final_answer = self.llm.predict(final_prompt)
        elif hasattr(self.llm, 'invoke'):
            res = self.llm.invoke(final_prompt)
            final_answer = res.content if hasattr(res, 'content') else str(res)
        elif hasattr(self.llm, 'call'):
             final_answer = self.llm.call(final_prompt)
        else:
            final_answer = "Error: LLM not callable."

        return final_answer

    def _run_single_crew(self, agent, task):
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False
        )
        return crew.kickoff()

    def _extract_json(self, text):
        try:
            if hasattr(text, 'raw'):
                text = text.raw

            if isinstance(text, list):
                return text
            if isinstance(text, dict):
                return [text]

            text = str(text)
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return []
        except:
            return []
