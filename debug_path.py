import sys
import langchain
import os

print(f"Python Executable: {sys.executable}")
print(f"LangChain Version: {langchain.__version__}")
print(f"LangChain Location: {os.path.dirname(langchain.__file__)}")