from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel
from datetime import datetime, date
from langchain.docstore.document import Document
import langchain
from enum import Enum
from cat.log import log
from cat.db import crud
import cat.factory.llm as llms

class LLMIds(Enum):
    OpenAIChat: str = 'LLMOpenAIChatConfig'
    use_default_llm: bool = True,
    LLOllama: str = 'LLMOllamaConfig'


class MySettings(BaseModel):
    query_llm_prefix: str = "Find the relevant information that should be embedded from the text retrieved from the source in a clean way, without any comments"
    use_default_llm: bool = True,
    llm_id: LLMIds = LLMIds.OpenAIChat

@plugin
def settings_schema():
    return MySettings.schema()


@hook
def before_rabbithole_splits_text(doc: Document, cat) -> Document:

    settings = cat.mad_hatter.plugins["LLMDAE-plugin"].load_settings()
    prompt = f"{settings['query_llm_prefix']}\n from source(${doc[0].metadata['source']}) \n text: {doc[0].page_content}"

    if settings['use_default_llm']:
        result = cat.llm(prompt)
    else:
        selected_llm_class = settings['llm_id']

        FactoryClass = getattr(llms, selected_llm_class)

        # obtain configuration and instantiate LLM
        selected_llm_config = crud.get_setting_by_name(name=selected_llm_class)
        try:
            llm = FactoryClass.get_llm_from_config(selected_llm_config["value"])
        except Exception as e:
            import traceback
            traceback.print_exc()
            llm = llms.LLMDefaultConfig.get_llm_from_config({})


        # Check if self._llm is a completion model and generate a response
        if isinstance(llm, langchain.llms.base.BaseLLM):
            result = llm(prompt, callbacks=[])

        # Check if self._llm is a chat model and call it as a completion model
        if isinstance(llm, langchain.chat_models.base.BaseChatModel):
            result = llm.call_as_llm(prompt, callbacks=[])


    doc[0].page_content = result

    return doc
