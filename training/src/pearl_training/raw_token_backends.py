"""Capture vLLM engine inputs/accepted tokens and released embedding tensors."""
from __future__ import annotations

from .raw_tokens import (TokenCaptureError, capture_enabled, disabled_capture_record,
                         modality_status, plain, recorded, tensor_inputs, writer)


class RecordedHelperCache:
    """Cache reuse is an input-lineage event, not a newly invented inference."""
    def __init__(self, delegate, identity):
        self.delegate, self.identity = delegate, identity
        if capture_enabled():
            writer("helper-cache-use")

    def invoke(self, payload, executor):
        if not capture_enabled():
            return self.delegate.invoke(payload, executor)
        with recorded("helper-cache-use", {"identity": self.identity, "payload": payload},
                      new_inference=False) as call:
            response, cached = self.delegate.invoke(payload, executor)
            call.finish(cache_hit=cached, source_inference_trace=response.get("pearl_raw_token_trace"),
                        historical_tokens_status=("available_via_source_trace" if response.get("pearl_raw_token_trace")
                                                  else "not_recorded_in_historical_fixed_input"))
            return response, cached


class VLLMTokenCapture:
    def __init__(self, llm):
        self.enabled = capture_enabled()
        if not self.enabled:
            return
        writer("vllm")
        self.active = None
        self.requests = {}
        core = llm.llm_engine.engine_core
        if not callable(getattr(core, "add_request", None)) or not callable(getattr(core, "get_output", None)):
            raise TokenCaptureError("This vLLM backend does not expose actual engine request/output capture points")
        original_add, original_output = core.add_request, core.get_output

        def add(request, *args, **kwargs):
            if self.active is None:
                raise TokenCaptureError("A vLLM request was submitted outside raw-token recording")
            if request.sampling_params is None or request.sampling_params.n != 1:
                raise TokenCaptureError("This recorder requires the fixed single-output sampling protocol")
            mm = [{"modality": f.modality, "identifier": f.identifier,
                   "offset": f.mm_position.offset, "length": f.mm_position.length}
                  for f in (request.mm_features or [])]
            call, prefix = self.active
            call.inputs(prefix=prefix, token_ids=request.prompt_token_ids, request_id=request.request_id,
                        multimodal_feature_positions=mm, eos_token_id=request.eos_token_id,
                        **modality_status())
            self.requests[request.request_id] = call
            return original_add(request, *args, **kwargs)

        def output(*args, **kwargs):
            result = original_output(*args, **kwargs)
            for row in result.outputs:
                call = self.requests.get(row.request_id)
                if call is None:
                    raise TokenCaptureError("Unrecorded vLLM engine output")
                call.unit(list(row.new_token_ids), unit_type="vllm_engine_decode_chunk",
                          request_id=row.request_id, engine_timestamp=result.timestamp,
                          finish_reason=str(row.finish_reason) if row.finish_reason is not None else None,
                          stop_reason=row.stop_reason,
                          engine_events=[{"type": str(x.type), "timestamp": x.timestamp} for x in (row.events or [])])
            return result

        core.add_request, core.get_output = add, output

    def generate(self, llm, inputs, sampling, *, metadata, **kwargs):
        if not self.enabled:
            return (llm.generate([inputs], sampling_params=sampling, **kwargs),
                    disabled_capture_record())
        with recorded("vllm", metadata) as call:
            if self.active is not None:
                raise TokenCaptureError("Concurrent calls must use separate isolated services")
            self.active = (call, inputs.get("prompt"))
            try:
                outputs = llm.generate([inputs], sampling_params=sampling, **kwargs)
                if not call.units:
                    raise TokenCaptureError("No raw engine token outputs were captured")
                reference = call.finish(returned_outputs=[{
                    "request_id": row.request_id, "prompt_token_ids": row.prompt_token_ids,
                    "outputs": [{"index": x.index, "token_ids": list(x.token_ids), "text": x.text,
                                 "finish_reason": x.finish_reason, "stop_reason": x.stop_reason}
                                for x in row.outputs]} for row in outputs])
                return outputs, reference
            finally:
                self.active = None
                self.requests.clear()


def trace_embedder(embedder, identity):
    """Wrap this instance only, preserving released preprocessing and outputs."""
    if not capture_enabled():
        return embedder
    writer("embedding")
    original_process = embedder.process
    original_forward = embedder.forward
    original_template = embedder.processor.apply_chat_template
    active = {}

    def template(*args, **kwargs):
        result = original_template(*args, **kwargs)
        if active:
            active["prefix"] = result
        return result

    def forward(inputs):
        if "call" not in active:
            raise TokenCaptureError("Embedding inference ran without an active token record")
        tensor_inputs(active["call"], inputs, embedder.processor, prefix=active.get("prefix"))
        return original_forward(inputs)

    def process(inputs, normalize=True):
        with recorded("embedding", {"identity": identity, "wire_inputs": inputs,
                                     "normalize": normalize}) as call:
            active["call"] = call
            try:
                result = original_process(inputs, normalize=normalize)
                call.unit([], unit_type="embedding_forward", sampled_tokens_status="not_applicable_embedding_only")
                call.finish(returned_embeddings=plain(result))
                return result
            finally:
                active.clear()

    embedder.processor.apply_chat_template = template
    embedder.forward = forward
    embedder.process = process
    return embedder


def generate_hf(model, processor, inputs, *, messages, metadata, **kwargs):
    """Streamer receives actual prompt/output IDs directly from generate."""
    with recorded("hf-generation", metadata) as call:
        tensor_inputs(call, inputs, processor, messages=messages)

        class Streamer:
            def __init__(self):
                self.first = True

            def put(self, value):
                ids = plain(value)
                if self.first:
                    self.first = False
                    call.event("generation_initial_prefix_ids", actual_input_token_ids=ids)
                else:
                    call.unit(ids, unit_type="hf_generate_decode_step")

            def end(self):
                call.event("generation_stream_end")

        result = model.generate(**inputs, streamer=Streamer(), **kwargs)
        call.finish(returned_sequence_token_ids=plain(result))
        return result


def forward_hf(inputs, processor, function, *, messages, metadata):
    with recorded("hf-embedding", metadata) as call:
        tensor_inputs(call, inputs, processor, messages=messages)
        result = function()
        call.unit([], unit_type="embedding_forward", sampled_tokens_status="not_applicable_embedding_only")
        call.finish(returned_embeddings=plain(result))
        return result
