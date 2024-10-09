from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

class GradeDocument(BaseModel):
    '''
    Binary score for relevance check on retrieved documents.
    '''
    
    binary_score: str = Field(
        description="Document are relevant to the question, 'yes' or 'no'"
    )
    

structured_llm_grader = llm.with_structured_output(GradeDocument)

system = """You are a grader assessing the relevance of a retrieved document to a user question. \n
    If the document contains keyword or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score of 'yes' or 'no' to indicate whether the document is relevant to the question."""
    
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
    
)

retrieval_grader = grade_prompt | structured_llm_grader