# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-docstring, invalid-name
"""A GEMM schedule rule for GPU operators."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from tvm import tir
from tvm.ir import Range
from tvm.target import Target
from tvm.tir import IterVar, PrimExpr, Var
from tvm.tir.analysis import undefined_vars
from tvm.tir.schedule.schedule import BlockRV

from ..base import analysis, BlockInfo, IterInfo
from .base import GPUScheduleRule

import json
import os
import subprocess

import math

def is_dynamic_loop(i) -> bool:
    """Check if a given loop has a dynamic shape."""

    # Check if the extent is dynamic (tir.Var or symbolic expression)
    print(type(i))
    if isinstance(i, tir.Var):
        print("This loop has a dynamic shape (variable-based extent).")
        return True
    else:
        print("This loop has a no dynamic shape (expression-based extent).")
        return False

    return False


# stotastic schedule
def get_tiling_factor(target, sch, i, j, k):

    # generate tiling factors

    possible_i_factors = []
    possible_j_factors = []
    possible_k_factors = []

    loop_i = False
    loop_j = False
    loop_k = False

    if is_dynamic_loop(sch.get(i).extent):
        loop_i = True
    if is_dynamic_loop(sch.get(j).extent):
        loop_j = True
    if is_dynamic_loop(sch.get(k).extent):
        loop_k = True

    # if is_dynamic_loop(sch, i, j, k):
    #     print(f"dynamic shape in i: {i}, j : {j}, k: {k}")
    # else:
    #     print(f"NO Dynamic shape in i: {i}, j : {j}, k: {k}")

    best_i_factors = None
    best_j_factors = None
    best_k_factors = None
    best_latency = float('inf')

    for _ in range(10):
        i_factors = sch.sample_perfect_tile(loop = i, n = 2, max_innermost_factor = 8)
        j_factors = sch.sample_perfect_tile(loop = j, n = 2, max_innermost_factor = 8)
        k_factors = sch.sample_perfect_tile(loop = k, n = 2, max_innermost_factor = 8)

    trace = sch.trace
    decisions = trace.decisions  # Access the decisions dictionary
    for inst in trace.insts:  # Iterate through all instructions
        if inst.kind.name == "SamplePerfectTile":  # Filter for tiling operations
            decision = decisions.get(inst)  # Retrieve the decision for this instruction
            if decision:
                if inst.inputs[0] == i:
                    if str(decision) not in [str(d) for d in possible_i_factors]:  # Check for duplication
                        possible_i_factors.append(decision)
                elif inst.inputs[0] == j: 
                    if str(decision) not in [str(d) for d in possible_j_factors]:  # Check for duplication
                        possible_j_factors.append(decision)
                elif inst.inputs[0] == k:
                    if str(decision) not in [str(d) for d in possible_k_factors]:  # Check for duplication
                        possible_k_factors.append(decision)
            else:
                print(f"No decision found for loop {inst.inputs[0]}")
    
    i_true = sch.get(i).extent
    k_true = sch.get(k).extent
    j_true = sch.get(j).extent
    print(f"type i: {type(i_true)}, i: {i_true} possible_i_factors: {possible_i_factors}")
    print(f"type j: {type(j_true)}, j: {j_true} possible_j_factors: {possible_j_factors}")
    print(f"type k: {type(k_true)}, i: {k_true} possible_k_factors: {possible_k_factors}")
    print("-------------------")            
    #target.Target.current(allow_none=False)

    memory_shared_size, l2_cache_size = target_memory_size(target)

    pre_defined_factor = [[256,4]]
    if loop_i:
        possible_i_factors = pre_defined_factor
        i_true = 1024
    if loop_j:
        possible_j_factors = pre_defined_factor
        j_true = 1024
    if loop_k:
        possible_k_factors = pre_defined_factor
        k_true = 1024

    #pre_defined_factor = [[1, 1], [2, 2], [4, 4], [8, 4], [8, 8]]
    
    # #reset possible tiling factors for negative tiling
    # if has_negative_values(possible_i_factors):
    #     print("negative in i")
    #     possible_i_factors.clear()
    #     possible_i_factors = pre_defined_factor
        
    # if has_negative_values(possible_j_factors):
    #     print("negative in j")
    #     possible_j_factors.clear()
    #     possible_j_factors = pre_defined_factor
        
    # if has_negative_values(possible_k_factors):
    #     print("negative in k")
    #     possible_k_factors.clear()
    #     possible_k_factors = pre_defined_factor
        
    # print(type(possible_i_factors))
    # print(type(possible_j_factors))
    # print(type(possible_k_factors))
    latency = float('inf')
    for tile_i in possible_i_factors:
        for tile_j in possible_j_factors:
            for tile_k in possible_k_factors:
                print(f"tile i {tile_i}, tile_j {tile_j}, tile_k {tile_k}, i_true {int(i_true)}, j_true, {int(j_true)}, k_true {int(k_true)}")
                if tile_k[1]<=8:
                    if tile_j[1] <=8:
                        if tile_i[1] <= 4:
                            latency = run_ace(str(int(i_true)), str(int(j_true)), str(int(k_true)), str(tile_i[0]), str(tile_j[0]), str(tile_k[0]))
                    elif tile_j[1] <=4:
                        if tile_i[1] <=8:
                            latency = run_ace(str(int(i_true)), str(int(j_true)), str(int(k_true)), str(tile_i[0]), str(tile_j[0]), str(tile_k[0]))

                    #print(f"tiling factors: {tile_i, tile_j, tile_k}")
                    #run ace
                    
                    #latency = run_ace(str(int(i_true)), str(int(j_true)), str(int(k_true)), str(tile_i[0]), str(tile_j[0]), str(tile_k[0]))
                    if latency is not None and not math.isinf(latency):
                        print(latency)
                        if latency < best_latency:
                            best_i_factors = tile_i
                            best_j_factors = tile_j
                            best_k_factors = tile_k
                # else:
                #     print(f"Exlude this combination tile i {tile_i}, tile_j {tile_j}, tile_k {tile_k}, i_true {i_true}, j_true, {j_true}, k_true {k_true}")

                    #look for best tiling best on ace result
    #check for limit:
    
    print(f"memory shared/block {memory_shared_size} L2_cache_size_bytes {l2_cache_size}")  # This will show you all available keys
    print(f"best i, j, k factors {best_i_factors}, {best_j_factors}, {best_k_factors}")
    #print(f"Shared memory limit : {shared_mem_limit}")

    if best_i_factors is None or best_j_factors is None or best_k_factors is None:
        best_i_factors = [None, 8]
        best_j_factors = [None, 8]
        best_k_factors = [None, 4]

    return [None, best_i_factors[1]], [None, best_j_factors[1]], [None, best_k_factors[1]]

#check for negative tiling factor
def has_negative_values(factors):
    """Check if there are any negative values in a nested list of factors."""
    return any(value < 0 for factor in factors for value in factor)

def save_config_to_json(i, j, k, i_factor, j_factor, k_factor, file_path="/home/jjlab/ace/relations/config.json",pe_h=16, pe_w=16):
    """Save ACE configuration parameters into a JSON file."""
    config = {
        "i": i,
        "j": j,
        "k": k,
        "tiling_i": i_factor,
        "tiling_j": j_factor,
        "tiling_k": k_factor,
        "pe_h": pe_h,
        "pe_w": pe_w
    }

    # Write the config to a JSON file
    with open(file_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {file_path}")


#run ace
def run_ace(i, j, k, i_factor, j_factor, k_factor):
    """Run ACE and extract latency from the log file."""
    # Paths for configuration and log files
    
    ace_log_path = "/home/jjlab/ace/ace.log"
    

    print("save config.json")
    save_config_to_json(i, j, k, i_factor, j_factor, k_factor)
    # Run ACE command
    print("Running ACE...")
    try:
        # Run the run_ace.py script using subprocess
        result = subprocess.run(
            ["python3", "/home/jjlab/ace/run_ace.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check for errors
        if result.returncode != 0:
            print(f"Error running ACE:\n{result.stderr}")
            return None

        # Extract and print the output
        print(result.stdout)

    except FileNotFoundError:
        print("Error: run_ace.py not found or Python not installed properly.")
        return None

    latency = None
    # Extract latency from the log file
    if os.path.exists(ace_log_path):
        with open(ace_log_path, "r") as f:
            for line in f:
                if "Total Delay" in line:
                    latency = line.split()[3]  # Extract latency value
                    print(f"Latency: {latency}")
                    return float(latency)
    if latency is None:
        print("Error: 'Total Delay' not found in the log file.")
    else:
        print(f"latency {latency}")
    return None
# retrieve target_memory_size
def target_memory_size(target):
    return target.max_shared_memory_per_block, target.l2_cache_size_bytes

def _collect_producers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for producer in sch.get_producers(block):
        result.append(producer)
        result.extend(_collect_producers(sch, producer))
    return result


def _collect_consumers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for consumer in sch.get_consumers(block):
        result.append(consumer)
        result.extend(_collect_consumers(sch, consumer))
    return result


def auto_inline_producers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        producers = _collect_producers(sch, block)
        for producer in producers:
            try:
                sch.compute_inline(producer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        consumers = _collect_consumers(sch, block)
        for consumer in consumers:
            try:
                sch.compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        for consumer in consumers:
            try:
                sch.reverse_compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumer_chain(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    auto_inline_consumers(sch, block)
    remaining_consumers = sch.get_consumers(block)

    if len(remaining_consumers) != 0:
        # Some blocks have failed to be inlined to the producer cache-write stage.
        # This could be due to another producer block that has not been scheduled.
        for c in remaining_consumers:
            for p in sch.get_producers(c):
                if sch.get(p) != sch.get(block):
                    auto_inline_producers(sch, p)
                    sch.compute_inline(p)

        # Try inlining into the cache-write stage again, this time it should succeed.
        auto_inline_consumers(sch, block)

    msg = "There are some consumers of the cache-write stage that are not properly inlined."
    assert len(sch.get_consumers(block)) == 0, msg


class IterKind(Enum):
    """Iter kinds for GEMM-liked programs.
    We can simplify the computation to C[S, I, J] += A[S, I, K] * B[S, J, K],
    where `I, J, K` are fundamental axes for gemm and `S` represents all
    other spatial axes (e.g. batches)
    kIter_S: spatial axes
    kIter_I: I axes
    kIter_J: J axes
    kIter_K: K axes
    kIter_T: trivial axes (i.e. with extent 1)
    """

    kIter_S = 0
    kIter_I = 1
    kIter_J = 2
    kIter_K = 3
    kIter_T = 4


@dataclass
class IterTrait:
    kind: IterKind
    extent: PrimExpr


def _is_one(x: PrimExpr) -> bool:
    return isinstance(x, tir.IntImm) and x.value == 1


def make_iter_fusion_index_map(
    traits: List[IterTrait],
    kind_order: List[IterKind],
) -> tir.IndexMap:
    fused_iters: Dict[IterKind, PrimExpr] = {}
    input_iters: List[tir.Var] = []
    for i, trait in enumerate(traits):
        v_i = tir.Var(f"i{i}", trait.extent.dtype)
        input_iters.append(v_i)
        if trait.kind == IterKind.kIter_T:
            continue
        if trait.kind not in kind_order:
            raise ValueError(f"Unknown iter kind {trait.kind}")
        if trait.kind in fused_iters:
            fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
        else:
            fused_iters[trait.kind] = v_i

    final_indices: List[tir.PrimExpr] = [
        fused_iters.get(kind, tir.IntImm(traits[0].extent.dtype, 0)) for kind in kind_order
    ]

    return tir.IndexMap(input_iters, final_indices, None)


def detect_iter_traits(block: tir.Block) -> Optional[Tuple[List[IterTrait]]]:
    """Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K]

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    Returns
    -------
    traits : Optional[Tuple[List[IterTrait]]]
        The detected iter traits for axes in A, B and C. None if the block
        does not match the pattern.

    """

    if len(block.reads) != 2 or len(block.writes) != 1:
        return None

    def get_access_axes(region: List[Range]) -> Set[Var]:
        axes: Set[Var] = set()
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes = axes.union(set(undefined_vars(r.min)))
        return axes

    try:
        A_axes = get_access_axes(block.reads[0].region)
        B_axes = get_access_axes(block.reads[1].region)
        C_axes = get_access_axes(block.writes[0].region)
    except ValueError:
        return None

    traits: Dict[Var, IterTrait] = {}
    for iter_var in block.iter_vars:
        var = iter_var.var
        kind: IterKind
        if _is_one(iter_var.dom.extent):
            kind = IterKind.kIter_T
        elif iter_var.iter_type == iter_var.DataPar:
            if var in A_axes and var in B_axes and var in C_axes:
                kind = IterKind.kIter_S
            elif var in A_axes and var in C_axes:
                kind = IterKind.kIter_I
            elif var in B_axes and var in C_axes:
                kind = IterKind.kIter_J
            else:
                return None
        elif iter_var.iter_type == tir.IterVar.CommReduce:
            if var in A_axes and var in B_axes and var not in C_axes:
                kind = IterKind.kIter_K
            else:
                return None
        else:
            return None
        traits[var] = IterTrait(kind, iter_var.dom.extent)

    # A Gemm-kernel requires have I, J and K axes
    gemm_traits = {IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K}
    if {x.kind for x in traits.values()}.intersection(gemm_traits) != gemm_traits:
        return None

    A_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in A_axes]
    B_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in B_axes]
    C_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in C_axes]
    block_traits = [traits[i.var] for i in block.iter_vars]
    return A_traits, B_traits, C_traits, block_traits


def get_index_map(block: tir.Block) -> Optional[Tuple[tir.IndexMap, ...]]:
    """Get index maps for the block

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    Returns
    -------
    index_maps : Optional[Tuple[tir.IndexMap]]
        The index maps for the block, or None if the block is not a gemm-liked kernel
    """
    traits = detect_iter_traits(block)
    if traits is None:
        return None
    A_traits, B_traits, C_traits, block_traits = traits

    A_index_map = make_iter_fusion_index_map(
        A_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_K]
    )
    B_index_map = make_iter_fusion_index_map(
        B_traits, [IterKind.kIter_S, IterKind.kIter_J, IterKind.kIter_K]
    )
    C_index_map = make_iter_fusion_index_map(
        C_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J]
    )
    matmul_index_map = make_iter_fusion_index_map(
        block_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K]
    )

    return (
        matmul_index_map,
        A_index_map,
        B_index_map,
        C_index_map,
    )


def get_block_info(sch: tir.Schedule, block: tir.schedule.BlockRV) -> BlockInfo:
    def _iter_kind(loop: tir.IterVar) -> str:
        return {tir.IterVar.DataPar: "S", tir.IterVar.CommReduce: "R"}.get(loop.iter_type, "O")

    def _is_reduction_block(block: tir.schedule.BlockRV):
        for iter_var in sch.get(block).iter_vars:
            if _iter_kind(iter_var) == "R":
                return True
        return False

    return BlockInfo(
        name=sch.get(block).name_hint,
        iters=[
            IterInfo(
                kind=_iter_kind(iter_var),
                var=iter_var.var,
                dom=iter_var.dom.extent,
                loop_rv=loop_rv,
            )
            for loop_rv, iter_var in zip(sch.get_loops(block), sch.get(block).iter_vars)
        ],
        block_rv=block,
        reduction_block=_is_reduction_block(block),
    )


def get_reduction_blocks(sch, blocks) -> bool:
    # Get the main computation block
    def is_reduction(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks


def get_in_out_dtypes(block: tir.Block) -> Tuple[str]:
    """
    Detect In/Out data types for the given block based on the analysis if read/write buffers.
    """
    assert len(block.reads) > 0 and len(block.writes) > 0
    in_dtype = block.reads[0].buffer.dtype
    out_dtype = block.writes[0].buffer.dtype
    return (in_dtype, out_dtype)


def check_sm_version(arch: str) -> int:
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1


class MetalMatmul(GPUScheduleRule):
    """
    The schedule rule for Metal matmul computation.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        from tvm.tir.tensor_intrin.metal import (  # pylint: disable=import-outside-toplevel
            get_simdgroup_intrin_group,
        )

        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        # Step 0. Configs
        block_size_x: int = 16
        block_size_y: int = 16
        block_size_k: int = 32
        micro_size: int = 8
        warp_size: int = 32
        ty_len: int = 1
        tz_len: int = 4
        vector_size: int = 4

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        # Reindex first and than analyze the index map
        main_block = reduction_blocks[0]
        reindex_a = sch.reindex(main_block, ("read", 0))
        reindex_b = sch.reindex(main_block, ("read", 1))
        reindex_c = sch.reindex(main_block, ("write", 0))

        index_maps = get_index_map(sch.get(main_block))
        assert index_maps is not None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        sch.transform_layout(reindex_a, ("write", 0), a_index_map)
        sch.transform_layout(reindex_b, ("write", 0), b_index_map)
        sch.transform_layout(reindex_c, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                ty_len * block_size_x,
                tz_len * block_size_y,
                block_size_k,
            ],
        )

        # Step 3. Schedule matmul to use simdgroup intrinsics
        batch, i, j, k = sch.get_loops(main_block)
        bx, ty, i0, i1 = sch.split(i, [None, ty_len, block_size_x // micro_size, micro_size])
        by, tz, j0, j1 = sch.split(j, [None, tz_len, block_size_y // micro_size, micro_size])
        k0, k1, k2 = sch.split(k, [None, block_size_k // micro_size, micro_size])
        sch.reorder(bx, by, ty, tz, k0, k1, i0, j0, i1, j1, k2)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(batch, "blockIdx.z")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tz, "threadIdx.z")

        def fetch_to_shared(block, idx):
            block_read = sch.cache_read(block, idx, "shared")
            sch.compute_at(block_read, k0, preserve_unit_loops=True)
            fused = sch.fuse(*sch.get_loops(block_read)[-2:])
            _, _tz, _ty, _tx, vec = sch.split(fused, [None, tz_len, ty_len, warp_size, vector_size])

            sch.bind(_tz, "threadIdx.z")
            sch.bind(_ty, "threadIdx.y")
            sch.bind(_tx, "threadIdx.x")
            sch.vectorize(vec)

            return block_read

        a_g2s = fetch_to_shared(main_block, 0)
        b_g2s = fetch_to_shared(main_block, 1)

        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        # create read cache to load matrix from shared memory to wmma fragments
        A_simdgroup = sch.cache_read(main_block, 0, "metal.simdgroup")
        B_simdgroup = sch.cache_read(main_block, 1, "metal.simdgroup")
        sch.compute_at(A_simdgroup, k1)
        sch.compute_at(B_simdgroup, k1)

        C_simd2s = sch.cache_write(main_block, 0, "metal.simdgroup")
        C_s2g = sch.cache_write(C_simd2s, 0, "shared")
        sch.reverse_compute_at(C_simd2s, tz, preserve_unit_loops=True)
        sch.reverse_compute_at(C_s2g, by, preserve_unit_loops=True)

        intrin_group = get_simdgroup_intrin_group(
            load_scope="shared",
            store_scope="shared",
            dtype="float16",
            trans_a=False,
            trans_b=True,
        )
        sch.transform_layout(B_simdgroup, ("write", 0), lambda s, i, j: (s, j, i))

        def tensorize_block(block: tir.schedule.BlockRV, intrin: str):
            *_, i, j = sch.get_loops(block)
            io, ii = sch.split(i, [None, micro_size])
            jo, ji = sch.split(j, [None, micro_size])
            sch.reorder(io, jo, ii, ji)
            sch.tensorize(ii, intrin)

        C_init = sch.decompose_reduction(main_block, k0)
        tensorize_block(A_simdgroup, intrin_group["load_a"])
        tensorize_block(B_simdgroup, intrin_group["load_b"])
        tensorize_block(C_simd2s, intrin_group["store"])
        tensorize_block(C_init, intrin_group["init"])

        *_, i, j, k = sch.get_loops(main_block)
        sch.tensorize(i, intrin_group["compute"])

        auto_inline_consumer_chain(sch, C_s2g)
        fused = sch.fuse(*sch.get_loops(C_s2g)[-2:])
        _, _tz, _ty, _tx, vec = sch.split(fused, [None, tz_len, ty_len, warp_size, vector_size])
        sch.bind(_tz, "threadIdx.z")
        sch.bind(_ty, "threadIdx.y")
        sch.bind(_tx, "threadIdx.x")
        sch.vectorize(vec)

        return sch


class MatmulTensorization(GPUScheduleRule):
    """
    The schedule rule for float16 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_wmma_intrin_group,
        )

        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        # tensor core intrinsic size
        micro_size_x = 16
        micro_size_y = 16
        micro_size_k = 16

        warp_size = 32
        vector_size = 4

        
        

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        # Reindex first and than analyze the index map
        main_block = reduction_blocks[0]
        reindex_a = sch.reindex(main_block, ("read", 0))
        reindex_b = sch.reindex(main_block, ("read", 1))
        reindex_c = sch.reindex(main_block, ("write", 0))

        index_maps = get_index_map(sch.get(main_block))
        assert index_maps is not None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        sch.transform_layout(reindex_a, ("write", 0), a_index_map)
        sch.transform_layout(reindex_b, ("write", 0), b_index_map)
        sch.transform_layout(reindex_c, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)


        batch_0, i_0, j_0, k_0 = sch.get_loops(main_block)
        print(f"batch0 {sch.get(batch_0).extent}, i0 {sch.get(i_0).extent}, j0 {sch.get(j_0).extent}, k0 {sch.get(k_0).extent}")
        get_tiling_factor(target, sch, i_0, j_0, k_0)
        #i_factors, j_factors, k_factors = 
        # i_factors, j_factors, k_factors = (
        #     [None, 1, 4, 2],
        #     [1, None, 4, 2],
        #     [None, 4],
        # )
        i_factors, j_factors, k_factors = (
            [None, 8],
            [None, 4],
            [None, 8],
         )

        num_ty = int(i_factors[1])
        x_pad_factor = int(i_factors[1])
        y_pad_factor = int(j_factors[1])
        k_pad_factor = int(k_factors[1])
        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                micro_size_x * x_pad_factor,
                micro_size_y * y_pad_factor,
                micro_size_k * k_pad_factor,
            ],
        )

        # Step 3. Schedule matmul to use tensor core
        block = main_block

        batch, i, j, k = sch.get_loops(block)
        
        print(f"batch {sch.get(batch).extent}, i {sch.get(i).extent}, j {sch.get(j).extent}, k {sch.get(k).extent}")
        # inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_x])
        j, j_inner = sch.split(j, factors=[None, micro_size_y])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)


        block_inner = block
        block_outer = sch.blockize(i_inner)

        
        print("Using FP16 tensorisation")
        i0, i1 = sch.split(i, factors=i_factors)
        j0, j1 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, k_factors)
        if target.arch.startswith("sm_") and int(target.arch[-2:]) > 75:
            sch.annotate(k0, "software_pipeline_order", [0, 3, 1, 4, 5, 2, 6])
            sch.annotate(k0, "software_pipeline_stage", [0, 0, 0, 0, 0, 1, 1])
            sch.annotate(k1, "software_pipeline_order", [0, 1, 2])
            sch.annotate(k1, "software_pipeline_stage", [0, 0, 1])

        sch.reorder(i0, j0, k0, i1, j1, k1)

        #block_idx = sch.fuse(i0, j0) #fused to mimic algorithm
        #block_idy = sch.fuse(i1, j1) # fused to mimic
        #thread_idy = sch.fuse(j2, i2)
        sch.bind(batch, "blockIdx.z")
        sch.bind(i1, "blockIdx.x")
        sch.bind(j0, "blockIdx.y")
        #sch.bind(thread_idy, "threadIdx.y")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared.dyn")
            sch.compute_at(block_read, k0)
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

            _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])

            sch.bind(f_2, "threadIdx.x")
            sch.bind(f_1, "threadIdx.y")
            sch.vectorize(f_3)

            sch.storage_align(block_read, 0, axis=-2, factor=16, offset=8)
            sch.annotate(block_read, "tir.manifest_shared_memory_local_stage", 1)
            sch.annotate(block_read, "double_buffer_scope", 0)
            return block_read

        a_g2s = fetch_to_shared(block_outer, 0, 2)
        b_g2s = fetch_to_shared(block_outer, 1, 2)

        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        # create read cache to load matrix from shared memory to wmma fragments
        A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
        B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")
        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        accumulator_shared_to_global = sch.cache_write(block_outer, 0, "shared.dyn")
        sch.storage_align(accumulator_shared_to_global, 0, -2, 16, 4)

        store = sch.cache_write(block_outer, 0, "wmma.accumulator")
        sch.reverse_compute_at(store, k1)
        sch.reverse_compute_at(accumulator_shared_to_global, k1)

        # split the store loop to match hardware intrinsic pattern
        i, j = sch.get_loops(store)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)

        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # Tensorization by hardware intrinsics
        intrin_group = get_wmma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype="float16",
            out_dtype="float32",
            trans_b=True,
        )

        try:
            i, j = sch.get_loops(A_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_a"])

            i, j = sch.get_loops(B_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_b"])
        except:  # pylint: disable=bare-except
            return None

        # Try to tensorize the init, store and compute block with f16 or f32 intrinsics
        tensorize_success: bool = False

        def tensorize_init_store_compute():
            sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

        try:
            tensorize_init_store_compute()
            tensorize_success = True
        except:  # pylint: disable=bare-except
            intrin_group = get_wmma_intrin_group(
                load_scope="shared.dyn",
                store_scope="shared.dyn",
                in_dtype="float16",
                out_dtype="float16",
                trans_b=True,
            )

        if not tensorize_success:
            try:
                tensorize_init_store_compute()
                tensorize_success = True
            except:  # pylint: disable=bare-except
                return None
        auto_inline_consumer_chain(sch, accumulator_shared_to_global)

        fused = sch.fuse(*sch.get_loops(accumulator_shared_to_global)[-2:])
        _, f1, f2 = sch.split(fused, factors=[None, warp_size, vector_size])
        sch.bind(f1, "threadIdx.x")
        sch.vectorize(f2)

        return sch if tensorize_success else None


class MatmulInt8Tensorization(GPUScheduleRule):
    """
    The schedule rule for int8 tensor core matmul computation.
    func with attr 'dlight.do_not_tensorize' will not be tensorized.
    """

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_wmma_intrin_group,
        )

        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        if "dlight.do_not_tensorize" in func.attrs.keys():
            return None

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None

        # Start Schedule
        # Step 0. Get schedule config.
        # NOTE: we can analyze the config by the hardware spec in the future

        # tensor core intrinsic size
        micro_size_x = 16
        micro_size_y = 16
        micro_size_k = 16

        warp_size = 32
        vector_size = 4

        print("Using int8 Tensor core")
        i_factors, j_factors, k_factors = (
            [None, 1, 4, 2],
            [1, None, 4, 2],
            [None, 1],
        )

        num_ty = i_factors[2] * j_factors[2]
        x_pad_factor = i_factors[2] * i_factors[3]
        y_pad_factor = j_factors[2] * j_factors[3]
        k_pad_factor = k_factors[1]

        # Step 1. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        # Reindex first and than analyze the index map
        main_block = reduction_blocks[0]
        reindex_a = sch.reindex(main_block, ("read", 0))
        reindex_b = sch.reindex(main_block, ("read", 1))
        reindex_c = sch.reindex(main_block, ("write", 0))

        index_maps = get_index_map(sch.get(main_block))
        assert index_maps is not None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        sch.transform_layout(reindex_a, ("write", 0), a_index_map)
        sch.transform_layout(reindex_b, ("write", 0), b_index_map)
        sch.transform_layout(reindex_c, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 2. Padding for dynamic shape kernels
        sch.pad_einsum(
            main_block,
            [
                1,
                micro_size_x * x_pad_factor,
                micro_size_y * y_pad_factor,
                micro_size_k * k_pad_factor,
            ],
        )

        # Step 3. Schedule matmul to use tensor core
        block = main_block

        batch, i, j, k = sch.get_loops(block)

        # inner loops for tensor core computation
        i, i_inner = sch.split(i, factors=[None, micro_size_x])
        j, j_inner = sch.split(j, factors=[None, micro_size_y])
        k, k_inner = sch.split(k, factors=[None, micro_size_k])

        sch.reorder(i, j, k, i_inner, j_inner, k_inner)

        block_inner = block
        block_outer = sch.blockize(i_inner)

        i0, i1, i2, i3 = sch.split(i, factors=i_factors)
        j0, j1, j2, j3 = sch.split(j, factors=j_factors)
        k0, k1 = sch.split(k, k_factors)
        if target.arch.startswith("sm_") and int(target.arch[-2:]) > 75:
            sch.annotate(k0, "software_pipeline_order", [0, 3, 1, 4, 5, 2, 6])
            sch.annotate(k0, "software_pipeline_stage", [0, 0, 0, 0, 0, 1, 1])
            sch.annotate(k1, "software_pipeline_order", [0, 1, 2])
            sch.annotate(k1, "software_pipeline_stage", [0, 0, 1])

        sch.reorder(i0, j0, i1, j1, j2, i2, k0, k1, i3, j3)

        block_idx = sch.fuse(i0, j0)
        block_idy = sch.fuse(i1, j1)
        thread_idy = sch.fuse(j2, i2)
        sch.bind(batch, "blockIdx.z")
        sch.bind(block_idx, "blockIdx.x")
        sch.bind(block_idy, "blockIdx.y")
        sch.bind(thread_idy, "threadIdx.y")

        def fetch_to_shared(block, idx, ndim):
            block_read = sch.cache_read(block, idx, "shared.dyn")
            sch.compute_at(block_read, k0)
            fused = sch.fuse(*sch.get_loops(block_read)[-ndim:])

            _, f_1, f_2, f_3 = sch.split(fused, factors=[None, num_ty, warp_size, vector_size])

            sch.bind(f_2, "threadIdx.x")
            sch.bind(f_1, "threadIdx.y")
            sch.vectorize(f_3)

            sch.storage_align(block_read, 0, axis=-2, factor=16, offset=0)
            sch.annotate(block_read, "tir.manifest_shared_memory_local_stage", 1)
            sch.annotate(block_read, "double_buffer_scope", 0)
            return block_read

        a_g2s = fetch_to_shared(block_outer, 0, 2)
        b_g2s = fetch_to_shared(block_outer, 1, 2)

        auto_inline_producers(sch, a_g2s)
        auto_inline_producers(sch, b_g2s)

        # create read cache to load matrix from shared memory to wmma fragments
        A_mat = sch.cache_read(block_outer, 0, "wmma.matrix_a")
        B_mat = sch.cache_read(block_outer, 1, "wmma.matrix_b")
        sch.compute_at(A_mat, k1)
        sch.compute_at(B_mat, k1)

        # create write cache to store matrix from wmma fragments to shared memory and global memory
        accumulator_shared_to_global = sch.cache_write(block_outer, 0, "shared.dyn")
        sch.storage_align(accumulator_shared_to_global, 0, -2, 16, 4)

        store = sch.cache_write(block_outer, 0, "wmma.accumulator")
        sch.reverse_compute_at(store, thread_idy)
        sch.reverse_compute_at(accumulator_shared_to_global, thread_idy)

        # split the store loop to match hardware intrinsic pattern
        i, j = sch.get_loops(store)[-2:]
        i0, i1 = sch.split(i, factors=[None, 16])
        j0, j1 = sch.split(j, factors=[None, 16])
        sch.reorder(i0, j0, i1, j1)

        block_init_c = sch.decompose_reduction(block_outer, k0)
        block_init_c_inner = sch.get_child_blocks(block_init_c)[0]

        # Tensorization by hardware intrinsics
        intrin_group = get_wmma_intrin_group(
            load_scope="shared.dyn",
            store_scope="shared.dyn",
            in_dtype="int8",
            out_dtype="int32",
            trans_b=True,
        )

        try:
            i, j = sch.get_loops(A_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_a"])

            i, j = sch.get_loops(B_mat)[-2:]
            i0, i1 = sch.split(i, factors=[None, 16])
            j0, j1 = sch.split(j, factors=[None, 16])
            sch.reorder(i0, j0, i1, j1)
            sch.unroll(i0)
            sch.unroll(j0)
            sch.tensorize(i1, intrin_group["load_b"])
        except:  # pylint: disable=bare-except
            return None

        def tensorize_init_store_compute():
            sch.tensorize(sch.get_loops(block_init_c_inner)[-2], intrin_group["init"])
            sch.tensorize(sch.get_loops(store)[-2], intrin_group["store"])
            sch.tensorize(sch.get_loops(block_inner)[-3], intrin_group["compute"])

        try:
            tensorize_init_store_compute()
        except:  # pylint: disable=bare-except
            return None

        auto_inline_consumer_chain(sch, accumulator_shared_to_global)

        fused = sch.fuse(*sch.get_loops(accumulator_shared_to_global)[-2:])
        _, f1, f2 = sch.split(fused, factors=[None, warp_size, vector_size])
        sch.bind(f1, "threadIdx.x")
        sch.vectorize(f2)

        return sch


class Matmul(GPUScheduleRule):
    """The schedule rule for matmul-like computation"""

    @dataclass
    class Config:
        block_size_x: int = 8
        block_size_y: int = 8
        vthread_x: int = 1
        vthread_y: int = 1
        micro_size_x: int = 4
        micro_size_y: int = 4
        micro_size_k: int = 8
        vector_size: int = 1
        unroll: int = 256  # 0 means no unroll
        use_shared: bool = True
        storage_align: bool = False
        inner_x: bool = False

    def get_configs(self, target: Target) -> Config:
        """Get the schedule config for the target"""
        if target.kind.name == "cuda" or target.kind.name == "rocm":
            return Matmul.Config(
                block_size_x=8,
                block_size_y=16,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=4,
                micro_size_y=4,
                micro_size_k=16,
                vector_size=2,
                unroll=256,
                use_shared=True,
                storage_align=True,
                inner_x=False,
            )
        elif target.kind.name == "opencl" and (
            ("android" in str(target.host)) or ("adreno" in str(target.attrs))
        ):
            return Matmul.Config(
                block_size_x=32,
                block_size_y=8,
                vthread_x=1,
                vthread_y=1,
                micro_size_x=8,
                micro_size_y=2,
                micro_size_k=16,
                vector_size=8,
                unroll=4,
                use_shared=False,
                storage_align=False,
                inner_x=True,
            )
        else:
            return Matmul.Config()

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        sch = tir.Schedule(func)
        config = self.get_configs(target)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        
        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None
        
        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)

        main_block_info = get_block_info(sch, main_block)
        iter_infos = main_block_info.iters
        if not get_index_map(block_stmt):
            return None
        
        # Checks if it's a inner reduction by getting the last matrix's inner Index
        def is_inner_reduction(block_stmt, iter_infos):
            end_it = block_stmt.reads[-1].region[-1].min
            return {it.var: it.kind for it in iter_infos}.get(end_it, "O") == "R"

        if (
            target.kind.name == "opencl"
            and (("android" in str(target.host)) or ("adreno" in str(target.attrs)))
        ) and not is_inner_reduction(block_stmt, iter_infos):
            ret = self.sch_outer_reduction(sch, config, main_block, blocks)
            if ret is not None:
                return ret

        # Step 0. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
        # Reindex first and than analyze the index map
        reindex_a = sch.reindex(main_block, ("read", 0))
        reindex_b = sch.reindex(main_block, ("read", 1))
        reindex_c = sch.reindex(main_block, ("write", 0))

        index_maps = get_index_map(sch.get(main_block))
        assert index_maps is not None
        matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps

        sch.transform_layout(reindex_a, ("write", 0), a_index_map)
        sch.transform_layout(reindex_b, ("write", 0), b_index_map)
        sch.transform_layout(reindex_c, ("read", 0), c_index_map)
        sch.transform_block_layout(main_block, matmul_index_map)

        # Step 1. Check Tensor Core support
        # Tensorization config:
        # If any value of I, J, K is fixed and less than this threshold,
        # tensorization rule will not be applied.
        minimal_tensorize_threshold = 64
        block_stmt = sch.get(main_block)
        
        if target.kind.name == "cuda" and check_sm_version(target.arch) >= 70:
            apply_tensorization: bool = True
            # the batch dimension is not taken into consideration.
            for item_var in block_stmt.iter_vars[1:]:
                extent = item_var.dom.extent
                if isinstance(extent, tir.expr.IntImm):
                    if extent.value <= minimal_tensorize_threshold:
                        apply_tensorization = False
            if apply_tensorization:
                # Analyze read/write buffers and choose correct tensorizer: int8 or fp16.
                in_dtype, out_dtype = get_in_out_dtypes(block_stmt)
                tensorize_sch = None
                if in_dtype == "int8" and out_dtype == "int32":
                    tensorize_sch = MatmulInt8Tensorization().apply(func, target, _)
                elif in_dtype == "float16" and out_dtype in ["float16", "float32"]:
                    tensorize_sch = MatmulTensorization().apply(func, target, _)
                if tensorize_sch is not None:
                    return tensorize_sch
        elif target.kind.name == "metal":
            try:
                return MetalMatmul().apply(func, target, _)
            except:  # pylint: disable=bare-except
                pass

        print(f"Using General tensorisation")
        # Step 2. Schedule matmul
        y_kernel_size = config.vthread_y * config.block_size_y * config.micro_size_y
        x_kernel_size = config.vthread_x * config.block_size_x * config.micro_size_x
        if config.inner_x:
            sch.pad_einsum(
                main_block,
                [1, y_kernel_size, x_kernel_size, config.micro_size_k],
            )
            batch, y, x, k = sch.get_loops(main_block)
        else:
            sch.pad_einsum(
                main_block,
                [1, x_kernel_size, y_kernel_size, config.micro_size_k],
            )
            batch, x, y, k = sch.get_loops(main_block)
        by, vy, ty, yi = sch.split(
            y, [None, config.vthread_y, config.block_size_y, config.micro_size_y]
        )
        bx, vx, tx, xi = sch.split(
            x, [None, config.vthread_x, config.block_size_x, config.micro_size_x]
        )
        ko, ki = sch.split(k, factors=[None, config.micro_size_k])
        reordered_loops = [by, bx, vy, vx, ty, tx, ko, ki] + (
            [yi, xi] if config.inner_x else [xi, yi]
        )
        sch.reorder(*reordered_loops)
        by = sch.fuse(batch, by)
        sch.bind(bx, "blockIdx.x")
        sch.bind(by, "blockIdx.y")
        sch.bind(vy, "vthread.y")
        sch.bind(vx, "vthread.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        inner_loop = config.micro_size_x if config.inner_x else config.micro_size_y
        if inner_loop % config.vector_size == 0:
            _, v = sch.split(reordered_loops[-1], [None, config.vector_size])
            sch.vectorize(v)

        if config.unroll > 0:
            sch.annotate(tx, ann_key="pragma_auto_unroll_max_step", ann_val=config.unroll)
            sch.annotate(tx, ann_key="pragma_unroll_explicit", ann_val=1)

        l2g = sch.cache_write(main_block, 0, "local")
        sch.reverse_compute_at(l2g, tx, preserve_unit_loops=True)
        if config.micro_size_x % config.vector_size == 0:
            _, v = sch.split(sch.get_loops(l2g)[-1], [None, config.vector_size])
            sch.vectorize(v)

        if config.use_shared:

            def _cooperative_fetch(index, vec_len):
                block = sch.cache_read(main_block, index, "shared")
                num_loops = len(sch.get_loops(block))
                sch.compute_at(block, ko, preserve_unit_loops=True)
                loops = sch.get_loops(block)[-num_loops:]
                ty, tx, _, vec = sch.split(
                    sch.fuse(*loops),
                    factors=[config.block_size_y, config.block_size_x, None, vec_len],
                )
                sch.vectorize(vec)
                sch.bind(ty, "threadIdx.y")
                sch.bind(tx, "threadIdx.x")
                if config.storage_align:
                    sch.storage_align(block, 0, axis=1, factor=8, offset=vec_len)
                return block

            a_g2s = _cooperative_fetch(0, vec_len=config.vector_size)
            b_g2s = _cooperative_fetch(1, vec_len=config.vector_size)

            auto_inline_producers(sch, a_g2s)
            auto_inline_producers(sch, b_g2s)
        else:
            auto_inline_producers(sch, main_block)

        auto_inline_consumer_chain(sch, l2g)

        sch.decompose_reduction(main_block, ko)
        return sch

    def sch_outer_reduction(
        self,
        sch: tir.Schedule,
        config: Config,
        reduction_block: tir.schedule.BlockRV,
        blocks: List[tir.schedule.BlockRV],
    ) -> Optional[tir.Schedule]:

        """Get vectorization factor"""

        def get_max_factor(n, factors):
            factors = sorted(factors, reverse=True)
            for factor in factors:
                if n % factor == 0:
                    return factor
            return 1

        reduction_loops = sch.get_loops(reduction_block)
        if not len(reduction_loops) == 4:
            return None

        mb, ms, n, k = reduction_loops
        if not (
            isinstance(sch.get(n).extent, tir.IntImm)
            and isinstance(sch.get(mb).extent, tir.IntImm)
            and isinstance(sch.get(ms).extent, tir.Var)
        ):
            return None

        Threads_X, Threads_Y, VecSize, Unroll_M = (
            config.block_size_x,
            config.block_size_y,
            config.vector_size,
            config.unroll,
        )
        VecSize = min(get_max_factor(sch.get(n).extent // Threads_X, [1, 2, 4, 8]), VecSize)
        dequant_block = None
        matmul_block = reduction_block
        epilogue_block = None
        if blocks[-1] is not matmul_block:
            epilogue_block = blocks[-1]
        for blk in blocks[:-1]:
            if "dequantize" in sch.get(blk).name_hint:
                dequant_block = blk
            elif blk is not matmul_block:
                sch.compute_inline(blk)

        m = sch.fuse(mb, ms)

        sch.pad_einsum(matmul_block, [1, Threads_Y * Unroll_M, Threads_X * VecSize, 1])

        rmat_block, wmat_block = (
            sch.get_producers(matmul_block)[0],
            sch.get_consumers(matmul_block)[0],
        )
        mo, mi, mu = sch.split(m, [None, Threads_Y, Unroll_M])
        no, ni, nv = sch.split(n, [None, Threads_X, VecSize])
        k0, k1, k2, k3 = sch.split(k, [None, (Threads_X * VecSize) // 32, 4, 8])
        sch.reorder(no, mo, ni, mi, k0, k1, k2, k3, mu, nv)

        sch.compute_at(rmat_block, k0)
        if dequant_block is not None:
            sch.compute_at(dequant_block, k3)
        sch.reverse_compute_at(wmat_block, mi)
        sch.set_scope(rmat_block, 0, "shared")
        sch.set_scope(matmul_block, 0, "local")

        if dequant_block is not None:
            sch.set_scope(dequant_block, 0, "local")

        sch.bind(mo, "blockIdx.y")
        sch.bind(no, "blockIdx.x")
        sch.bind(mi, "threadIdx.y")
        sch.bind(ni, "threadIdx.x")
        sch.vectorize(sch.get_loops(matmul_block)[-1])
        if dequant_block is not None:
            sch.vectorize(sch.get_loops(dequant_block)[-1])

        # Co-operative Memory Fetch
        ro, rv = sch.split(sch.get_loops(rmat_block)[-1], [None, VecSize])
        sch.bind(ro, "threadIdx.x")
        sch.vectorize(rv)

        wv = sch.get_loops(wmat_block)[-1]
        sch.vectorize(wv)

        # Scale and Quant Cache
        if dequant_block is not None:
            qb = sch.cache_read(dequant_block, 0, "local")
            sb = sch.cache_read(dequant_block, 1, "local")
            sch.compute_at(sb, k1)
            sch.compute_at(qb, k2)
            sch.set_scope(sb, 0, "local")
            sch.set_scope(qb, 0, "local")
            sch.vectorize(sch.get_loops(qb)[-1])
            sch.vectorize(sch.get_loops(sb)[-1])

        if epilogue_block is not None:
            sch.reverse_compute_at(epilogue_block, mi, preserve_unit_loops=True)
            sch.set_scope(wmat_block, 0, "local")
            sch.compute_inline(wmat_block)
            sch.vectorize(sch.get_loops(epilogue_block)[-1])

        sch.decompose_reduction(matmul_block, k0)
        return sch