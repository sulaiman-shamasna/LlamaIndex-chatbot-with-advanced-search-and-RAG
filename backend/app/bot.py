from helpers import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from backend.app.call_tools import *

class Chat:
    def __init__(self, input_files: List[str]):
        self.vector_query_tool = VectorQueryTool(input_files=input_files)
        self.vector_tool = self.vector_query_tool.get_tool()
        
        self.get_summary = SummaryTool(nodes=self.vector_query_tool.nodes)
        self.summary_tool = self.get_summary.get_tool()
        
        self.response_handler = ResponseHandler()
    
    def process_query(self, query: str):
        response = self.response_handler.get_response([self.vector_tool, self.summary_tool], query)
        answer = self.response_handler.inspect_response(response)
        return answer


# if __name__ == "__main__":
#     input_files = ["transformers.pdf"]
#     processor = Chat(input_files=input_files)
#     answer = processor.process_query("Can you please summarize the abstract of the paper.")
#     print('_________________________________________')
#     print(answer)