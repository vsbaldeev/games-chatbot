import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning, module="langgraph")

from src.bot.app import main

main()
