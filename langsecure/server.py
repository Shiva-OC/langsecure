import json
import logging
import os.path
from typing import List, Optional
from nemoguardrails.server.datastore.datastore import DataStore
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from .shield import Langsecure
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()

app.rails_config_path = os.environ.get("POLICY_STORE_DIR", "")
app.llm_engine = os.environ.get("LLM_ENGINE", "openai")
app.llm_model = os.environ.get("LLM_MODEL", "gpt-3.5-turbo-instruct")

class MemoryStore(DataStore):
    """A datastore implementation using a simple dict."""

    def __init__(self):
        """Constructor."""
        self.data = {}

    def set(self, key: str, value: str):
        self.data[key] = value

    def get(self, key: str) -> Optional[str]:
        return self.data.get(key)

class RequestBody(BaseModel):
    config_id: Optional[str] = Field(
        default=os.getenv("DEFAULT_CONFIG_ID", "default"),
        description="The id of the configuration to be used. If not set, the default configuration will be used.",
    )
    thread_id: Optional[str] = Field(
        default=None,
        min_length=16,
        max_length=255,
        description="The id of an existing thread to which the messages should be added.",
    )
    messages: List[dict] = Field(
        default=None, description="The list of messages in the current conversation."
    )
    context: Optional[dict] = Field(
        default=None,
        description="Additional context data to be added to the conversation.",
    )

llm_rails_instances = {}
datastore = MemoryStore()

def _get_langsecure(config_id: str) -> Langsecure:
    """Returns the rails instance for the given config id."""
    if config_id in llm_rails_instances:
        return llm_rails_instances[config_id]

    langsecure = Langsecure(policy_store=config_id, llm_engine=app.llm_engine, llm_model=app.llm_model)
    llm_rails_instances[config_id] = langsecure

    return langsecure

def get_openai_response(message):
    return {"choices": [{"message": {"role": "assistant", "content": message}}]}

@app.post("/v1/chat/completions")
def chat_completion(body: RequestBody, request: Request):
    """Chat completion for the provided conversation."""

    log.info("Got request for config %s", body.config_id)

    try:
        langsecure = _get_langsecure(body.config_id)
    except ValueError as ex:
        log.exception(ex)
        return get_openai_response(f"Could not load the {body.config_id} guardrails configuration.")

    try:
        messages = body.messages
        if body.context:
            messages.insert(0, {"role": "context", "content": body.context})

        # If we have a `thread_id` specified, we need to look up the thread
        datastore_key = None

        if body.thread_id:
            if datastore is None:
                raise RuntimeError("No DataStore has been configured.")

            # We make sure the `thread_id` meets the minimum complexity requirement.
            if len(body.thread_id) < 16:
                return get_openai_response("The `thread_id` must have a minimum length of 16 characters.")

            # Fetch the existing thread messages. For easier management, we prepend
            # the string `thread-` to all thread keys.
            datastore_key = "thread-" + body.thread_id
            thread_messages = json.loads(datastore.get(datastore_key) or "[]")

            # And prepend them.
            messages = thread_messages + messages

        messages = [(message["role"], message["content"]) for message in messages]

        chat_template = ChatPromptTemplate.from_messages(messages)
        deny, bot_message = langsecure._input_enforcer(prompt=chat_template.format())
        if not deny:
            deny, bot_message = langsecure._output_enforcer(prompt=messages, answer=bot_message)

        if not deny:
            # If we're using threads, we also need to update the data before returning
            # the message.
            if body.thread_id:
                datastore.set(datastore_key, json.dumps(messages + [bot_message]))    

    except Exception as ex:
        log.exception(ex)
        bot_message = "Internal server error."
    
    return get_openai_response(bot_message)