import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="langgraph")

from src.bot import main

main()
