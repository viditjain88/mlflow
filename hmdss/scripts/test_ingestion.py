import os
from crewai_tools import PDFSearchTool

def test_pdf_tool():
    pdf_path = "hmdss/data/policies/surgical_policy.pdf"

    # Configure the tool to use a local embedding model if possible.
    # CrewAI uses ChromaDB by default. To use a custom embedding model,
    # we need to check how to pass it.
    # According to CrewAI docs (general knowledge), we can pass `config`.
    # Let's try passing the embedder configuration.

    # Note: In recent CrewAI versions, `embedder` config is passed in `config`.
    # But let's try just initializing it. If it fails due to missing OpenAI key, we know we need config.

    try:
        # We need to set the OpenAI API key to something dummy if we are mocking or using local models,
        # but the tool might try to validate it.
        os.environ["OPENAI_API_KEY"] = "sk-proj-dummy"

        # To use local embeddings, we often need to specify it.
        # Let's try to see if we can use the `config` parameter.
        # This is a guess based on common patterns. If it fails, I'll inspect the error.

        tool = PDFSearchTool(
            pdf=pdf_path,
            config=dict(
                llm=dict(
                    provider="ollama", # or google, openai, azure
                    config=dict(
                        model="gemma:2b",
                        # base_url="http://localhost:11434" # hypothetical
                    ),
                ),
                embedder=dict(
                    provider="huggingface",
                    config=dict(
                        model="sentence-transformers/all-MiniLM-L6-v2",
                    ),
                ),
            )
        )

        # The tool might ingest upon first run or init.
        print("Tool initialized.")

    except Exception as e:
        print(f"Error initializing tool: {e}")

if __name__ == "__main__":
    test_pdf_tool()
