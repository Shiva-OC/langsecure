"""Arg pack components."""

from typing import Any, Callable, Dict, Optional
import concurrent.futures
import asyncio
import inspect

from llama_index.core.base.query_pipeline.query import (
    InputKeys,
    OutputKeys,
    QueryComponent,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.query_pipeline import QueryPipeline
from langsecure.factory import implements
from langsecure import Langsecure

class StopComponent(QueryComponent):
    """Stop component.

    When encountered in a pipeline, the execution stops post this component.
    """

    message: str = "Pipeline execution terminated."
    callback_manager: Any = None

    def _validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def validate_component_inputs(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return input

    def _validate_component_outputs(self, output: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(output["output"], str):
            raise ValueError(f"Output is not a string.")
        return output

    def set_callback_manager(self, callback_manager: Any) -> None:
        self.callback_manager = callback_manager

    def _run_component(self, **kwargs: Any) -> Any:
        """Run component."""
        msg = kwargs['message']

        if self.callback_manager:
            streaming_handler = None
            for handler in self.callback_manager.handlers:
                if hasattr(handler, "IDENTIFIER") and handler.IDENTIFIER == "streaming_handler":
                    streaming_handler = handler
                    break

            if streaming_handler and streaming_handler._target_streamer:
                async def streamer(response):
                    # Optionally send a metadata first if required, empty dict as in original code
                    streaming_handler._target_streamer.websocket_streamer.queue.put_nowait({})

                    await streaming_handler.on_stream_start()
                    await streaming_handler.on_new_token(response)
                    await streaming_handler.on_stream_end()

                async def async_stream(prompt):
                    await streamer(prompt)

                # Run async streaming in a sync context
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is None:
                    asyncio.run(async_stream(msg))
                else:
                    loop.run_until_complete(async_stream(msg))

        return {"output": msg}

    async def _arun_component(self, **kwargs: Any) -> Any:
        return self._run_component(**kwargs)

    @property
    def input_keys(self) -> InputKeys:
        return InputKeys.from_keys({"message"})

    @property
    def output_keys(self) -> OutputKeys:
        return OutputKeys.from_keys({"output"})


orig_get_next_module_keys = QueryPipeline.get_next_module_keys

@implements('llama_index.core.query_pipeline.query.QueryPipeline')
class LI_QueryPipeline(Langsecure):
    def shield(self, runnable: Any) -> Any:
        if runnable.dag.has_node("stop_component"):
            runnable.dag.remove_node("stop_component")

        self._parent = runnable
        self._parent_callables = {
            name: func for name, func in inspect.getmembers(runnable, predicate=inspect.ismethod)
        }
        self._parent.__class__.get_next_module_keys = self._get_next_module_keys
        return self._parent

    def _get_next_module_keys(self, run_state):
        if 'stop_component' in run_state.executed_modules:
            # stop the pipeline execution here
            return []

        next_stages = orig_get_next_module_keys(self._parent, run_state)
        for stage in next_stages:
            if stage == "input":
                for module_key, module_input in run_state.all_module_inputs.items():
                    if module_key == stage:
                        deny, deny_message = self._input_enforcer(module_input['question'])
                        if deny:
                            stop_component = StopComponent(message=deny_message)
                            run_state.all_module_inputs['stop_component'] = {"message": deny_message}
                            if "stop_component" not in self._parent.module_dict:
                                self._parent.add("stop_component", stop_component)
                                # Set the callback manager on the newly added stop_component
                                stop_component.set_callback_manager(self._parent.callback_manager)
                            return ["stop_component"]
            if stage == "stop_component":
                # post stop component, just return empty list
                return []
        return next_stages
