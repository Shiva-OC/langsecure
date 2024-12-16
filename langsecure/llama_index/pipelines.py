"""Arg pack components."""

from typing import Any, Callable, Dict, Optional
import concurrent.futures
from llama_index.core.base.query_pipeline.query import (
    InputKeys,
    OutputKeys,
    QueryComponent,
)
from llama_index.core.bridge.pydantic import Field
import asyncio

class StopComponent(QueryComponent):
    """Stop  component.

    When encountered in a pipeline, the execution stops post this component.

    """

    message: str = "Pipeline execution terminated."
    callback_manager: Any = None
    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs during run_component."""
        raise NotImplementedError

    def validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component inputs."""
        return input

    def _validate_component_outputs(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate component outputs."""
        # make sure output value is a list
        if not isinstance(output["output"], str):
            raise ValueError(f"Output is not a string.")
        return output

    def set_callback_manager(self, callback_manager: Any) -> None:
        """Set callback manager."""
        self.callback_manager = callback_manager

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        msg = kwargs['message']
        if self.callback_manager:
            for handler in self.callback_manager.handlers:
                if hasattr(handler, "IDENTIFIER"):
                    if handler.IDENTIFIER == "streaming_handler":
                        streaming_handler = handler
            async def streamer(response):
                if streaming_handler and streaming_handler._target_streamer:
                    streaming_handler._target_streamer.websocket_streamer.queue.put_nowait({})

                await streaming_handler.on_stream_start()
                await streaming_handler.on_new_token(response)
                await streaming_handler.on_stream_end()

            async def async_stream(prompt):
                response = await streamer(prompt)
                return response

            def stream_sync_to_async(prompt):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                result = loop.run_until_complete(async_stream(prompt))
                return result
            async def wait_for_future():
                result = await loop.run_in_executor(None, future.result)
                return result
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(stream_sync_to_async, msg)
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError as e:
                if str(e).startswith('There is no current event loop in thread'):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                else:
                    raise
            loop.run_until_complete(wait_for_future())
        return {"output": msg}

    async def _arun_component(self, **kwargs: Any) -> Any:
        """Run component (async)."""
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        """Input keys."""
        return InputKeys.from_keys({"message"})

    @property
    def output_keys(self) -> OutputKeys:
        """Output keys."""
        return OutputKeys.from_keys({"output"})


import inspect
from langsecure.factory import implements
from langsecure import Langsecure
from llama_index.core.query_pipeline import QueryPipeline

orig_get_next_module_keys = QueryPipeline.get_next_module_keys
@implements('llama_index.core.query_pipeline.query.QueryPipeline')
class LI_QueryPipeline(Langsecure):
    def shield(self, runnable: Any) -> Any:
        if runnable.dag.has_node("stop_component"):
            runnable.dag.remove_node("stop_component")

        self._parent = runnable
        self._parent_callables = {name: func for name, func in inspect.getmembers(runnable, predicate=inspect.ismethod)}
        self._parent.__class__.get_next_module_keys = self._get_next_module_keys
        return self._parent


    def _get_next_module_keys(self, run_state):
        
        if 'stop_component' in run_state.executed_modules:
            #stop the pipeline execution here
            return []

        next_stages = orig_get_next_module_keys(self._parent, run_state)
        for stage in next_stages:
            if stage == "input":
                for module_key, module_input in run_state.all_module_inputs.items():
                    if module_key == stage:
                        deny, deny_message = self._input_enforcer(module_input['question'])
                        if deny == True:
                            stop_component = StopComponent(message=deny_message)
                            #Execute a stop stage and return back to the caller
                            run_state.all_module_inputs['stop_component'] = {"message": deny_message}
                            if "stop_component" not in self._parent.module_dict:
                                self._parent.add("stop_component", stop_component)
                            return ["stop_component"]
            if stage == "stop_component":
                #post stop component, just return empty list
                return []
        return next_stages           
