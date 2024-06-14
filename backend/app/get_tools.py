
from helpers import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()


from llama_index.core import SimpleDirectoryReader

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex

from llama_index.core.vector_stores import MetadataFilters
from typing import List
from llama_index.core.vector_stores import FilterCondition
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI


from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool


documents = SimpleDirectoryReader(input_files=["transformers.pdf"]).load_data()

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)


vector_index = VectorStoreIndex(nodes)

def vector_query(
    query: str, 
    page_numbers: List[str]
) -> str:
    """Perform a vector search over an index.
    
    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.
    
    """

    metadata_dicts = [
        {"key": "page_label", "value": p} for p in page_numbers
    ]
    
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(
            metadata_dicts,
            condition=FilterCondition.OR
        )
    )
    response = query_engine.query(query)
    return response

vector_query_tool = FunctionTool.from_defaults(
    name="vector_tool",
    fn=vector_query
)

llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

summary_index = SummaryIndex(nodes)
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
summary_tool = QueryEngineTool.from_defaults(
    name="summary_tool",
    query_engine=summary_query_engine,
    description=(
        "Useful if you want to get a summary of the provided document!"
    ),
)

print('Is this the End?')

# Assuming you have already defined `llm`, `vector_query_tool`, and `summary_tool`
response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "Can you please summarize the abstract of the paper.", 
    verbose=True
)

# Inspecting the object to find all its attributes
attributes = [attr for attr in dir(response) if not attr.startswith("__")]

# Accessing and printing each attribute separately
for attr in attributes:
    try:
        value = getattr(response, attr)
        if not callable(value):  # Ensure it's not a method
            # print(f"{attr}: {value}", "\n", ' XXXXXXXXXXXX' "\n")
            print(f"{attr}:", "\n", ' XXXXXXXXXXXX' "\n")
            if attr == "response_gen":
                print('response_gen:', value)
            elif attr == "response":
                print('Response:', value)
    except AttributeError:
        print(f"Could not access attribute: {attr}")

