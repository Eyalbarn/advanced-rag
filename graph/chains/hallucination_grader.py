from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

class GradeHallucination(BaseModel):
    '''
    Binary score for hallucination present in generation answer.
    '''
    
    binary_score: bool = Field(
        description="Document are relevant to the question, 'yes' or 'no'"
    )
    
    
structured_llm_grader = llm.with_structured_output(GradeHallucination)

system = """You are a grader assessing whether an LLM  generation is grounded in / supported by a set of retrieved facts.\n
Give a binary score of 'yes' or 'no'. 'yes' means that the answer is grounded in / supported by the set of facts. 'no' means that the answer is not grounded in / supported by the set of facts.
"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader