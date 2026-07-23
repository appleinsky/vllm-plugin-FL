# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/compilation/compiler_interface.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Ascend-specific compiler interface for vllm-plugin-FL.

This module provides the CompilerInterface subclass used when Ascend graph
compilation (npugraph_ex / torchair) is enabled.  By default vllm-plugin-FL
falls back to eager mode on NPU, so this class is only instantiated when the
user explicitly enables `ascend_compilation_config.enable_npugraph_ex`.
"""

from __future__ import annotations

import logging
import copy
import functools
import os
from collections.abc import Callable
from hashlib import sha256
from typing import Any, cast

import torch
import torch.fx as fx
from torch._dynamo.backends.common import aot_autograd
from torch._inductor.compile_fx import graph_returns_tuple, make_graph_return_tuple
from torch._inductor.decomposition import select_decomp_table
from torch.fx import GraphModule
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.config import VllmConfig
from vllm.config.utils import Range


logger = logging.getLogger(__name__)

COMPILATION_PASS_KEYS = (
    "graph_fusion_manager",  # vllm-ascend pass key.
    "post_grad_custom_post_pass",  # vLLM default platform pass key.
)


def _get_ascend_compilation_config(vllm_config: VllmConfig | None) -> dict[str, Any]:
    if vllm_config is None:
        return {}
    return (vllm_config.additional_config or {}).get("ascend_compilation_config", {})


def _config_get(config: dict[str, Any], name: str, default: Any = None) -> Any:
    return config.get(name, default)


def compile_fx(
    graph: GraphModule,
    example_inputs: list,
    inner_compile: Callable,
    decompositions: dict,
) -> Callable:
    recursive_compile_fx = functools.partial(
        compile_fx, inner_compile=inner_compile, decompositions=decompositions
    )

    if not graph_returns_tuple(graph):
        return make_graph_return_tuple(graph, example_inputs, recursive_compile_fx)
    return aot_autograd(fw_compiler=inner_compile)(graph, example_inputs)


def _get_pass_manager(compiler_config: dict[str, Any]) -> Callable[[fx.GraphModule], fx.GraphModule]:
    for key in COMPILATION_PASS_KEYS:
        pass_manager = compiler_config.get(key)
        if pass_manager is not None:
            return pass_manager

    for value in compiler_config.values():
        if callable(value):
            return value

    def noop_pass_manager(graph: fx.GraphModule) -> fx.GraphModule:
        return graph

    return noop_pass_manager


def fusion_pass_compile(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    compile_range: Range,
    key: str | None = None,
) -> tuple[Callable | None, Any | None]:
    def compile_inner(graph, example_inputs):
        current_pass_manager = _get_pass_manager(compiler_config)
        graph = current_pass_manager(graph)
        return graph

    decompositions = select_decomp_table()

    compiled_fn = compile_fx(
        graph=graph,
        example_inputs=example_inputs,
        inner_compile=compile_inner,
        decompositions=decompositions,
    )

    return compiled_fn, None


def _compute_decode_cudagraph_batch_sizes(vllm_config: VllmConfig) -> list[int]:
    num_spec_tokens = (
        vllm_config.speculative_config.num_speculative_tokens
        if vllm_config.speculative_config
        else 0
    )
    uniform_decode_query_len = num_spec_tokens + 1
    max_num_tokens = vllm_config.scheduler_config.max_num_seqs * uniform_decode_query_len
    return [
        x
        for x in vllm_config.compilation_config.cudagraph_capture_sizes
        if max_num_tokens >= x >= uniform_decode_query_len
    ]


def _configure_backend(
    config: Any,
    ascend_compilation_config: dict[str, Any],
    vllm_config: VllmConfig,
    process_kwargs_options: Callable | None = None,
) -> None:
    enable_static_kernel = _config_get(
        ascend_compilation_config, "enable_static_kernel", False
    )
    if enable_static_kernel and "LOCAL_WORLD_SIZE" not in os.environ:
        actual_local_world_size = (
            vllm_config.parallel_config.local_world_size
            * vllm_config.parallel_config.data_parallel_size_local
        )
        os.environ["LOCAL_WORLD_SIZE"] = str(actual_local_world_size)
        logger.info(
            "Setting LOCAL_WORLD_SIZE=%d for static kernel (local_world_size=%d * data_parallel_size_local=%d).",
            actual_local_world_size,
            vllm_config.parallel_config.local_world_size,
            vllm_config.parallel_config.data_parallel_size_local,
        )

    if process_kwargs_options is not None:
        options: dict[str, Any] = {
            "force_eager": True,
            "inplace_pass": False,
            "clone_input": False,
            "clone_output": False,
        }
        if enable_static_kernel:
            logger.info(
                "enable_static_kernel is enabled, static shape kernel will be used to accelerate aclgraph execution."
            )
            options["static_kernel_compile"] = True
            options["_vllm_aclnn_static_kernel_sym_range"] = (
                _compute_decode_cudagraph_batch_sizes(vllm_config)
            )
        process_kwargs_options(config, {"options": options})
    else:
        config.mode = "reduce-overhead"
        config.debug.run_eagerly = True
        config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
        if enable_static_kernel:
            logger.info(
                "enable_static_kernel is enabled, static shape kernel will be used to accelerate aclgraph execution."
            )
            config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
            config.experimental_config.aclgraph._aclnn_static_shape_kernel_sym_value_range = (
                _compute_decode_cudagraph_batch_sizes(vllm_config)
            )


def _wrap_fake_inputs(example_inputs: list[Any]) -> list[Any]:
    from torch._guards import detect_fake_mode

    current_fake_mode = detect_fake_mode()
    if current_fake_mode is None:
        return example_inputs
    return [
        current_fake_mode.from_tensor(inp)
        if (
            isinstance(inp, torch.Tensor)
            and hasattr(inp, "fake_mode")
            and inp.fake_mode is not current_fake_mode
        )
        else inp
        for inp in example_inputs
    ]


def npugraph_ex_compile(
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    vllm_config: VllmConfig,
    ascend_compilation_config: dict[str, Any],
    compile_range: Range,
    key: str | None = None,
    cache_dir: str | None = None,
) -> tuple[Callable | None, Any | None]:
    logger.info(
        "Entering npugraph_ex_compile: key=%s cache_dir=%s compile_range=%s "
        "enable_npugraph_ex=%s enable_static_kernel=%s",
        key,
        cache_dir,
        compile_range,
        _config_get(ascend_compilation_config, "enable_npugraph_ex", False),
        _config_get(ascend_compilation_config, "enable_static_kernel", False),
    )
    try:
        import npugraph_ex as nge

        cache_path = os.path.join(cache_dir, key) if (cache_dir and key) else None
        logger.info(
            "Using npugraph_ex backend for Ascend compilation: key=%s cache_path=%s",
            key,
            cache_path,
        )
        torch.npu.set_compile_mode(jit_compile=False)
        config = nge.CompilerConfig()
        try:
            from npugraph_ex.configs.compiler_config import _process_kwargs_options
        except ImportError:
            from npugraph_ex.configs.npugraphex_config import _process_kwargs_options

        _configure_backend(
            config,
            ascend_compilation_config,
            vllm_config,
            process_kwargs_options=_process_kwargs_options,
        )
        import npugraph_ex.npu_fx_compiler as nfx

        original_get_compiled_gm = nfx._NpuFxCompiler._get_compiled_gm

        def patched_get_compiled_gm(self, graph, example_inputs):
            compiled_gm = original_get_compiled_gm(self, graph, example_inputs)
            if cache_path:
                py_code = compiled_gm.get_code()
                if py_code:
                    if "triton_kernel_wrapper" in py_code:
                        logger.info(
                            "Skipping npugraph_ex cache for graph containing Triton kernels "
                            "(kernel_side_table indices are process-local): %s",
                            cache_path,
                        )
                    else:
                        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                        with open(cache_path, "w") as f:
                            f.write(py_code)
                        logger.info("Saved compiled graph to cache: %s", cache_path)
            return compiled_gm

        nfx._NpuFxCompiler._get_compiled_gm = patched_get_compiled_gm
        try:
            backend = nge.get_npu_backend(compiler_config=config)
            if not graph_returns_tuple(graph):
                compiled_fn = make_graph_return_tuple(graph, example_inputs, backend)
            else:
                compiled_fn = backend(graph, example_inputs)
        finally:
            nfx._NpuFxCompiler._get_compiled_gm = original_get_compiled_gm
        return compiled_fn, (key, cache_path)
    except ImportError:
        import torchair

        logger.info(
            "npugraph_ex is unavailable; using torchair backend for Ascend compilation: key=%s",
            key,
        )
        torch.npu.set_compile_mode(jit_compile=False)
        config = torchair.CompilerConfig()
        _configure_backend(config, ascend_compilation_config, vllm_config)
        backend = torchair.get_npu_backend(compiler_config=config)
        if not graph_returns_tuple(graph):
            compiled_fn = make_graph_return_tuple(graph, example_inputs, backend)
        else:
            compiled_fn = backend(graph, example_inputs)
        return compiled_fn, None


class AscendCompiler(CompilerInterface):
    """Ascend compiler interface for graph fusion and npugraph_ex backends."""

    name = "AscendCompiler"

    def compute_hash(self, vllm_config: VllmConfig) -> str:
        self.vllm_config = vllm_config
        ascend_compilation_config = _get_ascend_compilation_config(vllm_config)

        import torch_npu

        factors = {
            "torch_npu_version": torch_npu.__version__,
            "enable_npugraph_ex": _config_get(
                ascend_compilation_config, "enable_npugraph_ex", False
            ),
            "enable_static_kernel": _config_get(
                ascend_compilation_config, "enable_static_kernel", False
            ),
        }
        logger.info("AscendCompiler hash factors: %s", factors)
        return sha256(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]

    def initialize_cache(self, cache_dir, disable_cache=False, prefix=""):
        self.cache_dir = cache_dir
        self.disable_cache = disable_cache

    def compile(
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Range,
        key: str | None = None,
    ) -> tuple[Callable | None, Any | None]:
        graph = copy.deepcopy(graph)
        example_inputs = _wrap_fake_inputs(example_inputs)

        ascend_compilation_config = _get_ascend_compilation_config(
            getattr(self, "vllm_config", None)
        )
        logger.info(
            "AscendCompiler.compile invoked: key=%s compile_range=%s "
            "env_VLLM_FL_ENABLE_NPUGRAPH_EX=%s config_enable_npugraph_ex=%s "
            "config_enable_static_kernel=%s",
            key,
            compile_range,
            os.environ.get("VLLM_FL_ENABLE_NPUGRAPH_EX"),
            _config_get(ascend_compilation_config, "enable_npugraph_ex", False),
            _config_get(ascend_compilation_config, "enable_static_kernel", False),
        )
        if _config_get(ascend_compilation_config, "enable_npugraph_ex", False):
            cache_dir = (
                None
                if getattr(self, "disable_cache", False)
                else getattr(self, "cache_dir", None)
            )
            logger.info(
                "enable_npugraph_ex is enabled, which will bring graph compilation optimization."
            )
            assert hasattr(self, "vllm_config")
            return npugraph_ex_compile(
                graph,
                example_inputs,
                compiler_config,
                self.vllm_config,
                ascend_compilation_config,
                compile_range,
                key,
                cache_dir,
            )
        return fusion_pass_compile(graph, example_inputs, compiler_config, compile_range, key)

    def load(self, handle, graph, example_inputs, graph_index, compile_range):
        key, path = handle
        if not path or not os.path.exists(path):
            logger.info(
                "npugraph_ex cache miss for key %s (file absent or not saved), recompiling",
                key,
            )
            graph = copy.deepcopy(graph)
            example_inputs = _wrap_fake_inputs(example_inputs)
            ascend_compilation_config = _get_ascend_compilation_config(
                getattr(self, "vllm_config", None)
            )
            assert hasattr(self, "vllm_config")
            compiled_fn, _ = npugraph_ex_compile(
                graph,
                example_inputs,
                {},
                self.vllm_config,
                ascend_compilation_config,
                compile_range,
                key,
                getattr(self, "cache_dir", None),
            )
            return compiled_fn

        from npugraph_ex.npu_fx_compiler import (
            _CompiledFxArtifacts,
            _CompiledFxGraph,
        )

        with open(path) as f:
            py_code = f.read()
        artifacts = _CompiledFxArtifacts()
        artifacts.py_code = py_code
        logger.info("Loaded npugraph_ex compilation cache from %s", path)
        compiled_fn = cast(
            Callable[..., Any], _CompiledFxGraph.load_artifacts(artifacts)
        )

        if not graph_returns_tuple(graph):
            inner_fn = compiled_fn

            def compiled_fn(*args, **kwargs):
                result = inner_fn(*args, **kwargs)
                if isinstance(result, (tuple, list)) and len(result) == 1:
                    return result[0]
                return result

        return compiled_fn
