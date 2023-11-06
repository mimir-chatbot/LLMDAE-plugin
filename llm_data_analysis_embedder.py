from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel
from datetime import datetime, date

class MySettings(BaseModel):
    query_llm_prefix: str = "Find the relevant information that should be embedded in this text:"


@plugin
def settings_schema():   
    return MySettings.schema()


