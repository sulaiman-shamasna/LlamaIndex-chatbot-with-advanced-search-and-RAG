from helpers import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

from helpers import get_openai_api_key
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.llms.openai import OpenAI
from typing import List

class VectorQueryTool:
    def __init__(self, input_files: List[str], chunk_size: int = 1024):
        self.documents = SimpleDirectoryReader(input_files=input_files).load_data()
        self.splitter = SentenceSplitter(chunk_size=chunk_size)
        self.nodes = self.splitter.get_nodes_from_documents(self.documents)
        self.vector_index = VectorStoreIndex(self.nodes)
    
    def vector_query(self, query: str, page_numbers: List[str]) -> str:
        """Perform a vector search over an index."""
        metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]
        query_engine = self.vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(metadata_dicts, condition=FilterCondition.OR)
        )
        response = query_engine.query(query)
        return response
    
    def get_tool(self):
        return FunctionTool.from_defaults(name="vector_tool", fn=self.vector_query)


class SummaryTool:
    def __init__(self, nodes):
        self.summary_index = SummaryIndex(nodes)
        self.summary_query_engine = self.summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
    
    def get_tool(self):
        return QueryEngineTool.from_defaults(
            name="summary_tool",
            query_engine=self.summary_query_engine,
            description="Useful if you want to get a summary of the provided document!"
        )


class ResponseHandler:
    def __init__(self, model="gpt-3.5-turbo", temperature=0):
        self.llm = OpenAI(model=model, temperature=temperature)
    
    def get_response(self, tools: List, query: str):
        response = self.llm.predict_and_call(tools, query, verbose=True)
        return response
    
    @staticmethod
    def inspect_response(response):
        attributes = [attr for attr in dir(response) if not attr.startswith("__")]
        for attr in attributes:
            try:
                value = getattr(response, attr)
                if not callable(value):
                    # print(f"{attr}:", "\n", 'XXXXXXXXXXXX', "\n")
                    # if attr == "response_gen":
                    #     print('response_gen:', value)
                    if attr == "response":
                        # print('Response:', value)
                        return f'Response: {value}'
            except AttributeError:
                print(f"Could not access attribute: {attr}")


# if __name__ == "__main__":
#     input_files = ["transformers.pdf"]
    
#     vector_query_tool = VectorQueryTool(input_files=input_files)
#     vector_tool = vector_query_tool.get_tool()
    
#     summary_tool = SummaryTool(nodes=vector_query_tool.nodes)
#     summary_tool_instance = summary_tool.get_tool()
    
#     response_handler = ResponseHandler()
#     response = response_handler.get_response([vector_tool, summary_tool_instance], "Can you please summarize the abstract of the paper.")
#     answer = response_handler.inspect_response(response)
#     print('_________________________________________')
#     print(answer)
