# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.toolkits import (
    AudioAnalysisToolkit,
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    WebToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.logger import set_log_level
import requests
import tempfile
import os
from typing import Dict, List, Any

from utils import OwlRolePlaying, run_society, DocumentProcessingToolkit

load_dotenv()

set_log_level(level="DEBUG")


class ArxivToolkit:
    """A toolkit for directly downloading and processing arXiv papers."""
    
    def __init__(self, document_model=None):
        self.document_model = document_model
        self.doc_toolkit = DocumentProcessingToolkit(model=document_model)
    
    def download_arxiv_paper(self, arxiv_url: str) -> str:
        """
        Download an arXiv paper from its URL.
        
        Args:
            arxiv_url (str): The URL of the arXiv paper. Can be abstract or PDF URL.
        
        Returns:
            str: Path to the downloaded PDF file or error message.
        """
        # Ensure the URL ends with .pdf
        if not arxiv_url.endswith('.pdf'):
            # Convert abstract URL to PDF URL if needed
            if 'arxiv.org/abs/' in arxiv_url:
                paper_id = arxiv_url.split('arxiv.org/abs/')[1].split('.')[0]
                arxiv_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            elif 'arxiv.org/pdf/' in arxiv_url and not arxiv_url.endswith('.pdf'):
                arxiv_url = f"{arxiv_url}.pdf"
        
        try:
            response = requests.get(arxiv_url, stream=True)
            response.raise_for_status()
            
            # Create a temporary file to store the PDF
            fd, temp_path = tempfile.mkstemp(suffix='.pdf')
            os.close(fd)
            
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return f"Successfully downloaded arXiv paper to {temp_path}"
        except Exception as e:
            return f"Error downloading arXiv paper: {str(e)}"
    
    def analyze_arxiv_paper(self, arxiv_url: str) -> str:
        """
        Download and analyze an arXiv paper.
        
        Args:
            arxiv_url (str): The URL of the arXiv paper.
        
        Returns:
            str: Analysis of the paper or error message.
        """
        download_result = self.download_arxiv_paper(arxiv_url)
        if "Successfully downloaded" in download_result:
            pdf_path = download_result.split("to ")[1]
            # Use the document toolkit to analyze the PDF
            text_content = self.doc_toolkit.extract_text_from_pdf(pdf_path)
            analysis = self.doc_toolkit.summarize_document(text_content)
            
            # Clean up the temporary file
            try:
                os.remove(pdf_path)
            except:
                pass
                
            return analysis
        else:
            return download_result
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the list of tools provided by this toolkit."""
        download_schema = {
            "name": "download_arxiv_paper",
            "description": "Download an arXiv paper from its URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_url": {
                        "type": "string",
                        "description": "The URL of the arXiv paper. Can be abstract or PDF URL."
                    }
                },
                "required": ["arxiv_url"]
            }
        }
        
        analyze_schema = {
            "name": "analyze_arxiv_paper",
            "description": "Download and analyze an arXiv paper",
            "parameters": {
                "type": "object",
                "properties": {
                    "arxiv_url": {
                        "type": "string",
                        "description": "The URL of the arXiv paper. Can be abstract or PDF URL."
                    }
                },
                "required": ["arxiv_url"]
            }
        }
        
        return [
            {"type": "function", "function": download_schema, "implementation": self.download_arxiv_paper},
            {"type": "function", "function": analyze_schema, "implementation": self.analyze_arxiv_paper}
        ]


def construct_society(question: str) -> OwlRolePlaying:
    r"""Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        OwlRolePlaying: A configured society of agents ready to address the question.
    """

    # Create models for different components
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
        "web": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
        "video": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
        "document": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O,
            model_config_dict={"temperature": 0},
        ),
    }

    # Create ArxivToolkit instance
    arxiv_toolkit = ArxivToolkit(document_model=models["document"])

    # Configure toolkits
    tools = [
        # Add the custom ArxivToolkit
        *arxiv_toolkit.get_tools(),
        # Keep other tools as fallbacks
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        *VideoAnalysisToolkit(model=models["video"]).get_tools(),
        *AudioAnalysisToolkit().get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_google,
        SearchToolkit().search_wiki,
        *ExcelToolkit().get_tools(),
        *WebToolkit(
            headless=False,
            web_agent_model=models["web"],
            planning_agent_model=models["planning"],
        ).get_tools(),
    ]

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    society = OwlRolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )

    return society


def main():
    r"""Main function to run the OWL system with an example question."""
    # Example research question
    question = "What is in the given link: https://ar5iv.org/pdf/2311.12983"

    # Construct and run the society
    society = construct_society(question)
    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")


if __name__ == "__main__":
    main()