from typing import Any, Dict

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
import os
import sys

# Add the root of your project to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)  # Use insert(0, ...) to make sure it's the first in the list
print(f"Current sys.path: {sys.path}")  # Debugging line

from graph.state import GraphState
from dotenv import load_dotenv
load_dotenv()


web_search_tool = TavilySearchResults(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    #use get method to get the value of the key "documents" from the state dictionary
    documents = state.get("documents",[])
    
   
    
    tavily_results = web_search_tool.invoke({"query": question})
    
    joined_tavily_results = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    
    # for tavily_result in tavily_results:
    #     #print(f"WEB SEARCH RESULT: {tavily_result}")
    #     print(f"WEB SEARCH RESULT CONTENT: {tavily_result['content']}")
        
    print(f"WEB SEARCH RESULTS: {joined_tavily_results}")
    web_results = Document(page_content=joined_tavily_results)
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}
    
if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents":None})

