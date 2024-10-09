from dotenv import load_dotenv
import os
import sys
# Add the root of your project to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)  # Use insert(0, ...) to make sure it's the first in the list
print(f"Current sys.path: {sys.path}")  # Debugging line

load_dotenv()
#print(f'The openai api key:{os.getenv("OPENAI_API_KEY")}')  # Check if the API key is being loaded
from pprint import pprint

from graph.chains.hallucination_grader import hallucination_grader, GradeHallucination

from graph.chains.retrieval_grader import GradeDocument, retrieval_grader
from graph.chains.generation import generation_chain
from graph.chains.router import question_router, RouteQuery
from ingestion import retriever

def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    docs_txt = docs[0].page_content
    
    res: GradeDocument = retrieval_grader.invoke(
        {"question": question, "document": docs_txt}
    )
    
    assert res.binary_score == "yes"
    
def test_retrival_grader_answer_no() -> None:
    question = "how to make pizza?"
    docs = retriever.invoke(question)
    docs_txt = docs[0].page_content
    
    res: GradeDocument = retrieval_grader.invoke(
        {"question": question, "document": docs_txt}
    )
    
    assert res.binary_score == "no"
    
def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)
    
def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    
    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    
    assert res.binary_score
    
def test_hallucination_grader_answer_no() -> None:
    question = "agent memory?"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    
    res: GradeHallucination = hallucination_grader.invoke(
        {"documents": docs, "generation": "In order to make pizza we need to start with the dough."}
    )
    
    print(res.binary_score)
    assert not res.binary_score
    
def test_router_to_vectorestore() -> None:
    question = "agent memory"
    res : RouteQuery = question_router.invoke({"question": question})

    assert res.datasource == "vectorstore"
    
def test_router_to_websearch() -> None:
    question = "how to make pizza?"
    res : RouteQuery = question_router.invoke({"question": question})

    assert res.datasource == "websearch"