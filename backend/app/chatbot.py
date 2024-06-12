from typing import List, Tuple, Any
from dotenv import load_dotenv
from helpers import get_openai_api_key
from llama_index.core import SimpleDirectoryReader, Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

class LlamaIndexBase:
    def __init__(self, input_files: List[str]) -> None:
        """
        Initialize the LlamaIndexBase class.

        :param input_files: List of input file paths
        """
        self.load_environment()
        self.OPENAI_API_KEY = self.get_api_key()
        self.documents = self.load_documents(input_files)
        self.nodes = self.split_documents(self.documents)
        self.setup_models()
        self.summary_index, self.vector_index = self.create_indices(self.nodes)
        self.summary_tool, self.vector_tool = self.create_tools(self.summary_index, self.vector_index)
        self.query_engine = self.create_query_engine([self.summary_tool, self.vector_tool])

    def load_environment(self) -> None:
        """
        Load environment variables from a .env file.
        """
        load_dotenv()

    def get_api_key(self) -> str:
        """
        Retrieve the OpenAI API key.

        :return: OpenAI API key
        """
        return get_openai_api_key()

    def load_documents(self, input_files: List[str]) -> List[Any]:
        """
        Load documents from the specified input files.

        :param input_files: List of input file paths
        :return: List of loaded documents
        """
        return SimpleDirectoryReader(input_files=input_files).load_data()

    def split_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into nodes using a sentence splitter.

        :param documents: List of documents
        :return: List of nodes
        """
        splitter = SentenceSplitter(chunk_size=1024)
        return splitter.get_nodes_from_documents(documents)

    def setup_models(self) -> None:
        """
        Set up the models for LLM and embedding.
        """
        Settings.llm = OpenAI(model="gpt-3.5-turbo")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

    def create_indices(self, nodes: List[Any]) -> Tuple[SummaryIndex, VectorStoreIndex]:
        """
        Create summary and vector indices from nodes.

        :param nodes: List of nodes
        :return: Tuple containing summary index and vector index
        """
        summary_index = SummaryIndex(nodes)
        vector_index = VectorStoreIndex(nodes)
        return summary_index, vector_index

    def create_tools(self, summary_index: SummaryIndex, vector_index: VectorStoreIndex) -> Tuple[QueryEngineTool, QueryEngineTool]:
        """
        Create tools for querying the summary and vector indices.

        :param summary_index: Summary index
        :param vector_index: Vector index
        :return: Tuple containing summary tool and vector tool
        """
        summary_query_engine = summary_index.as_query_engine(
            response_mode="tree_summarize",
            use_async=True,
        )
        vector_query_engine = vector_index.as_query_engine()

        summary_tool = QueryEngineTool.from_defaults(
            query_engine=summary_query_engine,
            description="Useful for summarization questions related to the given paper",
        )

        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description="Useful for retrieving specific context from the given paper.",
        )

        return summary_tool, vector_tool

    def create_query_engine(self, tools: List[QueryEngineTool]) -> RouterQueryEngine:
        """
        Create a query engine using the provided tools.

        :param tools: List of query engine tools
        :return: RouterQueryEngine instance
        """
        return RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(),
            query_engine_tools=tools,
            verbose=True
        )

class LlamaIndexChatbot(LlamaIndexBase):
    def __init__(self, input_files: List[str]) -> None:
        """
        Initialize the LlamaIndexChatbot class.

        :param input_files: List of input file paths
        """
        super().__init__(input_files)

    def get_response(self, user_input: str) -> str:
        """
        Get a response from the query engine based on user input.

        :param user_input: User input string
        :return: Response string
        """
        response = self.query_engine.query(user_input)
        return str(response)

    def get_metadata(self, user_input: str) -> str:
        """
        Get metadata from the query engine based on user input.

        :param user_input: User input string
        :return: Metadata string
        """
        response = self.query_engine.query(user_input)
        return "This will contain metadata"

    def get_summarizer_tool(self, user_input: str) -> str:
        """
        Get a response from the summarizer tool based on user input.

        :param user_input: User input string
        :return: Response string
        """
        response = self.summary_tool.query_engine.query(user_input)
        return str(response)

    def get_vector_tool(self, user_input: str) -> str:
        """
        Get a response from the vector tool based on user input.

        :param user_input: User input string
        :return: Response string
        """
        response = self.vector_tool.query_engine.query(user_input)
        return str(response)

# Example usage:
# if __name__ == "__main__":
#     chatbot = LlamaIndexChatbot(input_files=["transformers.pdf"])

#     response = chatbot.get_response("What is the summary of the document?")
#     print(response)

#     response = chatbot.get_response("How do agents share information with other agents?")
#     print(response)
