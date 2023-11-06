from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel
from datetime import datetime, date
from langchain.docstore.document import Document
from cat.log import log

class MySettings(BaseModel):
    query_llm_prefix: str = "Find the relevant information that should be embedded from the text retrieved from the source in a clean way, without any comments"


@plugin
def settings_schema():   
    return MySettings.schema()


@hook
def before_rabbithole_splits_text(doc: Document, cat) -> Document:

    settings = cat.mad_hatter.plugins["LLMDAE-plugin"].load_settings()

    result = cat.llm(f"{settings['query_llm_prefix']}"
            f"\n from source(${doc[0].metadata['source']})"
            f"\n text: {doc[0].page_content}")
    doc[0].page_content = result
    return doc