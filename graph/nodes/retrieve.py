from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever
import os
from dotenv import load_dotenv

load_dotenv()

def retrieve(state:GraphState) -> Dict[str, Any]:
    print("---Retrieve---")
    question = state["question"]
    
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

# if __name__ == "__main__":
#     state = {
#         "question": "explain the concept of token generation",
#         "generation": "",
#         "web_search": True,
#         "documents": []
#     }
#     #print the working directory
#     print(os.getcwd())
#     print(retrieve(state))
