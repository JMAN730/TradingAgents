import warnings

# langchain-core imports pydantic's v1 shim, which warns on Python 3.14+;
# harmless for this codebase (pydantic v2 models only) — silence the noise
# before cli.main pulls in langchain via tradingagents.
warnings.filterwarnings("ignore", message="Core Pydantic V1", category=UserWarning)
