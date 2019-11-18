import pytest
import torch

from softargmax import SoftArgmax2D, SoftArgmax3D


def allclose(result, expect):
    return all([r == e for r, e in zip(result, expect)])


def test_soft_argmax_2d_cpu():
    arr = torch.randn(2, 7, 32, 32).double().cpu()
    result = SoftArgmax2D()(arr)
    assert allclose(result.shape, (2, 7, 2))


def test_soft_argmax_3d_cpu():
    arr = torch.randn(2, 5, 25, 32, 32).double().cpu()
    result = SoftArgmax3D()(arr)
    assert allclose(result.shape, (2, 5, 3))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable.")
def test_soft_argmax_2d_gpu():
    arr = torch.randn(2, 7, 32, 32).double().cuda()
    result = SoftArgmax2D()(arr)
    assert result.device != "cpu"
    assert allclose(result.shape, (2, 7, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable.")
def test_soft_argmax_3d_gpu():
    arr = torch.randn(2, 5, 25, 32, 32).double().cuda()
    result = SoftArgmax3D()(arr)
    assert result.device != "cpu"
    assert allclose(result.shape, (2, 5, 3))
