# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.tensor_parallel.random import (
    CheckpointWithoutOutput,
    CudaRNGStatesTracker,
    checkpoint,
    colocated_encoder_rng_tracker,
    convert_cuda_rng_state,
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
    snapshot_colocated_encoder_rng_tracker,
)
from tests.unit_tests.test_utilities import Utils


def test_cuda_rng_states_tracker():
    rng_tracker = CudaRNGStatesTracker()
    rng_tracker.set_states({"state1": 1234})
    assert rng_tracker.get_states()["state1"] == 1234
    rng_tracker.reset()
    assert rng_tracker.get_states() == {}
    seed = 1111
    rng_tracker.add("state2", seed)
    with pytest.raises(Exception):
        assert rng_tracker.add("state3", seed)
    with pytest.raises(Exception):
        assert rng_tracker.add("state2", 111)
    assert rng_tracker.get_states()['state2'] is not None
    with pytest.raises(Exception):
        assert ()

    rng_tracker.fork("state2")
    torch.cuda.manual_seed(seed)
    rng_state = torch.cuda.get_rng_state()
    assert torch.equal(rng_tracker.get_states()['state2'], rng_state)


@pytest.mark.parametrize("use_cudagraphable_rng", [True, False])
def test_double_fork_cuda_rng_states_tracker(use_cudagraphable_rng):
    rng_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=use_cudagraphable_rng)
    rng_tracker.add("state1", 1234)
    rng_tracker.add("state2", 5678)
    randn_double_fork_1 = []
    randn_double_fork_2 = []
    with rng_tracker.fork("state1"):
        randn_double_fork_1.append(torch.randn(10, device="cuda"))
        with rng_tracker.fork("state2"):
            randn_double_fork_2.append(torch.randn(10, device="cuda"))
            with rng_tracker.fork("state1"):
                randn_double_fork_1.append(torch.randn(10, device="cuda"))
            randn_double_fork_2.append(torch.randn(10, device="cuda"))
        randn_double_fork_1.append(torch.randn(10, device="cuda"))
    if use_cudagraphable_rng:
        double_fork_state1 = rng_tracker.get_states()["state1"].get_state()
        double_fork_state2 = rng_tracker.get_states()["state2"].get_state()
    else:
        double_fork_state1 = rng_tracker.get_states()["state1"]
        double_fork_state2 = rng_tracker.get_states()["state2"]

    rng_tracker.reset()
    rng_tracker.add("state1", 1234)
    rng_tracker.add("state2", 5678)
    randn_single_fork_1 = []
    randn_single_fork_2 = []
    with rng_tracker.fork("state1"):
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
        randn_single_fork_1.append(torch.randn(10, device="cuda"))
    with rng_tracker.fork("state2"):
        randn_single_fork_2.append(torch.randn(10, device="cuda"))
        randn_single_fork_2.append(torch.randn(10, device="cuda"))
    if use_cudagraphable_rng:
        single_fork_state1 = rng_tracker.get_states()["state1"].get_state()
        single_fork_state2 = rng_tracker.get_states()["state2"].get_state()
    else:
        single_fork_state1 = rng_tracker.get_states()["state1"]
        single_fork_state2 = rng_tracker.get_states()["state2"]

    assert torch.equal(randn_double_fork_1[0], randn_single_fork_1[0])
    assert torch.equal(randn_double_fork_1[1], randn_single_fork_1[1])
    assert torch.equal(randn_double_fork_1[2], randn_single_fork_1[2])
    assert torch.equal(randn_double_fork_2[0], randn_single_fork_2[0])
    assert torch.equal(randn_double_fork_2[1], randn_single_fork_2[1])
    assert torch.equal(double_fork_state1, single_fork_state1)
    assert torch.equal(double_fork_state2, single_fork_state2)


def test_convert_cuda_rng_state():
    ## Get the default rng state
    torch.cuda.manual_seed(999)
    randn = torch.randn(10, device="cuda")
    rng_state = torch.cuda.get_rng_state()

    try:
        from megatron.core.extensions.transformer_engine import TECudaRNGStatesTracker
    except ImportError:
        TECudaRNGStatesTracker = None

    ## from non-graphable RNG to graphable RNG
    # get state from non-graphable RNG
    tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
    tracker.add("state1", 123)
    for i in range(3):
        with tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
    state = convert_cuda_rng_state(tracker.states_["state1"], to_graphable=True)
    rand_tensors = []
    for i in range(3):
        with tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
            rand_tensors.append(randn)

    # set state to local graph RNG
    cudagraphable_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=True)
    cudagraphable_tracker.set_states({"state1": state.clone_state()})
    for i in range(3):
        with cudagraphable_tracker.fork("state1"):
            randn = torch.randn(10, device="cuda")
            assert torch.equal(randn, rand_tensors[i])

    # set state to TE RNG
    if TECudaRNGStatesTracker is not None:
        te_tracker = TECudaRNGStatesTracker()
        te_tracker.set_states({"state1": state})
        for i in range(3):
            with te_tracker.fork("state1"):
                randn = torch.randn(10, device="cuda")
                assert torch.equal(randn, rand_tensors[i])

    ## from graphable RNG to non-graphable RNG
    # get state from graphable RNG
    cudagraphable_tracker = CudaRNGStatesTracker(use_cudagraphable_rng=True)
    cudagraphable_tracker.add("state2", 123)
    for i in range(3):
        with cudagraphable_tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
    state = convert_cuda_rng_state(cudagraphable_tracker.states_["state2"], to_graphable=False)
    rand_tensors = []
    for i in range(3):
        with cudagraphable_tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
            rand_tensors.append(randn)

    # set state to non-graphable RNG
    tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
    tracker.set_states({"state2": state})
    for i in range(3):
        with tracker.fork("state2"):
            randn = torch.randn(10, device="cuda")
            assert torch.equal(randn, rand_tensors[i])

    ## from TE RNG to non-graphable RNG
    if TECudaRNGStatesTracker is not None:
        # get state from TE RNG
        cudagraphable_tracker = TECudaRNGStatesTracker()
        cudagraphable_tracker.add("state3", 123)
        for i in range(3):
            with cudagraphable_tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
        state = convert_cuda_rng_state(cudagraphable_tracker.states_["state3"], to_graphable=False)
        rand_tensors = []
        for i in range(3):
            with cudagraphable_tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
                rand_tensors.append(randn)

        # set state to non-graphable RNG
        tracker = CudaRNGStatesTracker(use_cudagraphable_rng=False)
        tracker.set_states({"state3": state})
        for i in range(3):
            with tracker.fork("state3"):
                randn = torch.randn(10, device="cuda")
                assert torch.equal(randn, rand_tensors[i])

    ## After all tests, check if the default rng state is still the same.
    rng_state_final = torch.cuda.get_rng_state()
    assert torch.equal(rng_state, rng_state_final)


def test_model_parallel_cuda_manual_seed():
    Utils.initialize_model_parallel(4, 2)
    model_parallel_cuda_manual_seed(0, force_reset_rng=True)
    rng_tracker = get_cuda_rng_tracker()
    assert rng_tracker.get_states()['model-parallel-rng'] is not None
    Utils.destroy_model_parallel()


def test_colocated_encoder_rng_tracker_isolation():
    """Task 5.12: the encoder's snapshotted tracker survives a later re-seed.

    This is the whole point of snapshotting a TRACKER instead of registering one extra
    named state: ``model_parallel_cuda_manual_seed`` resets the tracker and re-seeds
    every named state on each call, so the component seeded LAST (the backbone) owns
    "model-parallel-rng". After the snapshot, the context manager must hand out the
    ENCODER's version of that name - the one used by every default-name fork inside the
    encoder forward, attention dropout above all.
    Task 5.12：快照下来的 encoder tracker 必须扛得住之后的重新播种。这正是"快照整个
    tracker"而非"多注册一个命名状态"的意义：``model_parallel_cuda_manual_seed`` 每次调用
    都 reset tracker 并重播所有命名状态，因此**最后**播种的组件（backbone）占有
    "model-parallel-rng"。快照之后，上下文管理器必须交出 **encoder** 那一版——它才是
    encoder 前传里所有默认名 fork（首先是 attention dropout）实际使用的那条流。
    """
    Utils.initialize_model_parallel(1, 1)
    try:
        # Seed as the encoder component would, then freeze the tracker.
        # 先按 encoder 组件的方式播种，然后冻结 tracker。
        model_parallel_cuda_manual_seed(123, force_reset_rng=True)
        with get_cuda_rng_tracker().fork():
            encoder_reference_sample = torch.randn(64, device="cuda", dtype=torch.float32)
        model_parallel_cuda_manual_seed(123, force_reset_rng=True)
        snapshot_colocated_encoder_rng_tracker()

        # Re-seed as the backbone component would; this wipes and re-registers every
        # named state, so the live tracker no longer holds the encoder's version.
        # 再按 backbone 组件的方式重新播种：所有命名状态被清空并重播，活动 tracker 里
        # 已经不是 encoder 那一版了。
        model_parallel_cuda_manual_seed(456)
        with get_cuda_rng_tracker().fork():
            backbone_sample = torch.randn(64, device="cuda", dtype=torch.float32)
        assert not torch.equal(backbone_sample, encoder_reference_sample), (
            "the backbone re-seed must change the default-name stream; otherwise this "
            "test cannot tell the two trackers apart"
        )

        # Inside the context the default-name fork must resolve to the ENCODER's stream.
        # 在上下文内，默认名 fork 必须解析到 **encoder** 的流。
        with colocated_encoder_rng_tracker():
            with get_cuda_rng_tracker().fork():
                encoder_sample = torch.randn(64, device="cuda", dtype=torch.float32)
        assert torch.equal(encoder_sample, encoder_reference_sample), (
            "the colocated encoder tracker must keep the states seeded from the "
            "encoder's own coordinates, not the backbone's"
        )

        # The caller's tracker must be restored exactly on exit.
        # 退出后调用方的 tracker 必须被精确还原。
        with get_cuda_rng_tracker().fork():
            backbone_sample_after = torch.randn(64, device="cuda", dtype=torch.float32)
        assert not torch.equal(backbone_sample_after, encoder_sample), (
            "leaving the context must restore the caller's tracker"
        )
    finally:
        Utils.destroy_model_parallel()


def test_colocated_encoder_rng_tracker_advances_across_calls():
    """Task 5.12: the encoder stream keeps advancing instead of restarting each step.

    The context manager writes the advanced global CUDA RNG state back to the encoder
    side on exit, so two successive encoder forwards must not draw the same numbers.
    Task 5.12：encoder 的随机流必须持续前进，而不是每步都从快照重新开始。上下文管理器在
    退出时把推进后的全局 CUDA RNG 状态写回 encoder 一侧，因此连续两次 encoder 前传不能
    抽到相同的数。
    """
    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123, force_reset_rng=True)
        snapshot_colocated_encoder_rng_tracker()

        # The global stream is what the fused hidden dropout consumes (it does not fork).
        # 融合的 hidden dropout 消费的就是全局流（它不 fork）。
        with colocated_encoder_rng_tracker():
            first_step_sample = torch.randn(64, device="cuda", dtype=torch.float32)
        with colocated_encoder_rng_tracker():
            second_step_sample = torch.randn(64, device="cuda", dtype=torch.float32)
        assert not torch.equal(first_step_sample, second_step_sample), (
            "the encoder global RNG state must be carried over between forwards"
        )
    finally:
        Utils.destroy_model_parallel()


def test_checkpoint():
    def test_forward(*input):
        return input[0] + input[1]

    assert torch.equal(
        torch.ones(16) * 3, checkpoint(test_forward, None, torch.ones(16), torch.ones(16) * 2)
    )
    Utils.initialize_model_parallel()
    input1 = torch.ones((4, 4))
    checkpoint(test_forward, True, input1, torch.ones((4, 4)) * 2)
    assert torch.equal(torch.ones(input1.numel()).cuda(), input1)
    Utils.destroy_model_parallel()


def test_checkpoint_without_output():
    def normal_forward(input):
        x = torch.nn.functional.gelu(input)
        y = x * input
        return y

    def checkpoint_forward(input):
        checkpoint = CheckpointWithoutOutput()
        x = checkpoint.checkpoint(torch.nn.functional.gelu, input)
        y = x * input
        checkpoint.discard_output_and_register_recompute(y)
        return y

    Utils.initialize_model_parallel()

    input1 = torch.ones((4, 4))
    input1.requires_grad_(True)
    output1 = normal_forward(input1)
    input2 = torch.ones((4, 4))
    input2.requires_grad_(True)
    output2 = checkpoint_forward(input2)
    assert torch.equal(output1, output2)

    output1.backward(torch.ones((4, 4)), retain_graph=True)
    output2.backward(torch.ones((4, 4)), retain_graph=True)
    assert torch.equal(input1.grad, input2.grad)


class _ViewSavingLinear(torch.autograd.Function):
    """Saves view tensors in forward to mimic TE GroupedLinear-style backward inputs."""

    @staticmethod
    def forward(ctx, inp, weight):
        inp_2d = inp.reshape(-1, inp.shape[-1])
        inputmats = torch.tensor_split(inp_2d, 2, dim=0)
        ctx.save_for_backward(*inputmats, weight)
        ctx.input_shape = inp.shape
        out_2d = inp_2d.matmul(weight.t())
        return out_2d.reshape(*inp.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        *inputmats, weight = ctx.saved_tensors
        for inputmat in inputmats:
            if inputmat.numel() > 0 and inputmat.untyped_storage().size() == 0:
                raise RuntimeError("Saved view tensor points to an empty storage.")

        inp_2d = torch.cat(inputmats, dim=0)
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_input_2d = grad_output_2d.matmul(weight)
        grad_weight = grad_output_2d.t().matmul(inp_2d)
        grad_input = grad_input_2d.reshape(ctx.input_shape)
        return grad_input, grad_weight


def test_checkpoint_without_output_view_sharing_regression():
    def normal_forward(input_, weight):
        x = torch.nn.functional.gelu(input_)
        return _ViewSavingLinear.apply(x, weight)

    def checkpoint_forward(input_, weight):
        checkpoint = CheckpointWithoutOutput()
        x = checkpoint.checkpoint(torch.nn.functional.gelu, input_)
        y = _ViewSavingLinear.apply(x, weight)
        checkpoint.discard_output_and_register_recompute(y)
        return y

    Utils.initialize_model_parallel()
    try:
        input1 = torch.randn((3, 2, 8), requires_grad=True)
        weight1 = torch.randn((6, 8), requires_grad=True)

        input2 = input1.detach().clone().requires_grad_(True)
        weight2 = weight1.detach().clone().requires_grad_(True)

        output1 = normal_forward(input1, weight1)
        output2 = checkpoint_forward(input2, weight2)
        assert torch.allclose(output1, output2)

        grad = torch.randn_like(output1)
        output1.backward(grad, retain_graph=True)
        output2.backward(grad, retain_graph=True)
        assert torch.allclose(input1.grad, input2.grad)
        assert torch.allclose(weight1.grad, weight2.grad)
    finally:
        Utils.destroy_model_parallel()
