from langchain_community.document_loaders.base import BaseLoader

from langchain_core.documents import Document

from typing import Iterator

import os

class ObsFileLoader(BaseLoader):
    """Load Obsidian Markdown file.
    
    Args:
        file_path: Path to the file to load.
    
    """

    def __init__(self, file_path:str) -> None:
        self.file_path = file_path
    
    def lazy_load(self) -> Iterator[Document]:
        with open(self.file_path, encoding="utf-8") as file:
            
            text = file.read()
            filename = os.path.basename(file.name)
            print(f"{text} \n {filename}")
            yield Document(
                page_content=text,
                metadata={
                    "file_name": filename,
                }
            )
    
