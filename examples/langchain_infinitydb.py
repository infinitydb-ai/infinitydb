"""InfinityDB LangChain Adapter Example.

Importable and runnable directly for LangChain integration verification.
"""

from examples.langchain_integration import InfinityDBLangChainMemory, main
from infinitydb import InfinityDB

__all__ = ["InfinityDB", "InfinityDBLangChainMemory"]

if __name__ == "__main__":
    main()
