
from helpers import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()


# 2. Define a Simple Tool (to show how a tool calling works)
# from llama_index.core.tools import FunctionTool

# def add(x: int, y: int) -> int:
#     """Adds two integers together."""
#     return x + y

# def mystery(x: int, y: int) -> int: 
#     """Mystery function that operates on top of two numbers."""
#     return (x + y) * (x + y)

# add_tool = FunctionTool.from_defaults(fn=add)
# mystery_tool = FunctionTool.from_defaults(fn=mystery)

# # 3. Call and define the model
# from llama_index.llms.openai import OpenAI

# llm = OpenAI(model="gpt-3.5-turbo")
# response = llm.predict_and_call(
#     [add_tool, mystery_tool], 
#     "Tell me the output of the mystery function on 2 and 9", 
#     verbose=True
# )
# print(str(response))

# ------------------------------------------------

from llama_index.core import SimpleDirectoryReader
documents = SimpleDirectoryReader(input_files=["transformers.pdf"]).load_data()

# 5. Split into chunks 
from llama_index.core.node_parser import SentenceSplitter
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)

# print(nodes[0].get_content(metadata_mode="all"))

# ------------------------------------------------

from llama_index.core import VectorStoreIndex

vector_index = VectorStoreIndex(nodes)
query_engine = vector_index.as_query_engine(similarity_top_k=2)

# 7. Query the RAG pipeline via metadata filters
from llama_index.core.vector_stores import MetadataFilters

query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=MetadataFilters.from_dicts(
        [
            {"key": "page_label", "value": "2"}
        ]
    )
)

response = query_engine.query(
    "What are some high-level results of Attention mechanisms?", 
)

print(str(response))


print(10* '-------')

for n in response.source_nodes:
    print(n.metadata)


# ----------------------------------------------------

from typing import List
from llama_index.core.vector_stores import FilterCondition
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

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
response = llm.predict_and_call(
    [vector_query_tool], 
    "What doest the abstract section include in the first page.", 
    verbose=True
)

# ----------------------------------------

from llama_index.core import SummaryIndex
from llama_index.core.tools import QueryEngineTool

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

# Is this the end.
print('Is this the End?')
response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "List the contributers to this paper written on the first page, please!?", 
    verbose=True
)

# Is this the end.
print('Is this the End?')
response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "What's the summary of the results of this article", 
    verbose=True
)

# Is this the end.
print('Is this the End?')
response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "Can you please summarize the abstract of the paper.", 
    verbose=True
)