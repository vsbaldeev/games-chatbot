import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="langgraph")

from src.app.app import main

main()
