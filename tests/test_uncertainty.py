import pytest
import torch
from torch import nn

from nidavellir_tools.uncertainty import (
    mc_dropout_mode,
    mc_dropout_predict,
    select_class_uncertainty,
)


@pytest.mark.parametrize("dropout", [nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d])
def test_mode_restores_mixed_flags_and_rates(dropout):
    model = nn.Sequential(nn.BatchNorm3d(2), dropout(0.4)).train()
    model[0].eval()
    states = [m.training for m in model.modules()]
    with pytest.raises(RuntimeError), mc_dropout_mode(model):
        assert not model.training and not model[0].training and model[1].training
        assert model[1].p == 0.4
        raise RuntimeError("failure")
    assert [m.training for m in model.modules()] == states


@pytest.mark.parametrize("correction", [0, 1])
@pytest.mark.parametrize("shape", [(2, 3, 8, 8), (2, 3, 4, 8, 8)])
def test_statistics_match_stacked_samples(shape, correction):
    torch.manual_seed(7)
    model = nn.Sequential(nn.Dropout(0.5)).eval()
    result = mc_dropout_predict(
        model,
        torch.ones(shape),
        num_samples=8,
        correction=correction,
        output_transform=lambda x: x.softmax(dim=1),
        return_samples=True,
    )
    assert result.mean.shape == shape
    assert result.samples.shape == (8, *shape)
    assert not result.mean.requires_grad
    assert torch.any(result.std > 0)
    torch.testing.assert_close(result.mean, result.samples.mean(0))
    torch.testing.assert_close(result.variance, result.samples.var(0, correction=correction))
    torch.testing.assert_close(result.std, result.samples.std(0, correction=correction))
    torch.testing.assert_close(result.mean.sum(1), torch.ones_like(result.mean[:, 0]))
    assert not model[0].training


def test_batchnorm_buffers_unchanged_and_seed_repeatable():
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Dropout2d(0.5)).train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.ones(2, 3, 8, 8)
    torch.manual_seed(8)
    first = mc_dropout_predict(model, x, num_samples=5)
    torch.manual_seed(8)
    second = mc_dropout_predict(model, x, num_samples=5)
    torch.testing.assert_close(first.mean, second.mean)
    assert first.samples is None
    assert all(m.training for m in model.modules())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_callback_tuple_output_and_half_precision():
    model = nn.Dropout(0.5)
    result = mc_dropout_predict(
        model,
        (torch.ones(2, 1, 8).half(), 2),
        num_samples=3,
        predict_fn=lambda net, args: (net(args[0]) * args[1], "ignored"),
        output_transform=lambda outputs: outputs[0].sigmoid(),
    )
    assert result.mean.shape == (2, 1, 8)
    assert result.mean.dtype == torch.float32


@pytest.mark.parametrize("model", [nn.Identity(), nn.Dropout(0), nn.Dropout(1)])
def test_missing_stochastic_dropout_rejected(model):
    with pytest.raises(ValueError, match="supported dropout"):
        mc_dropout_predict(model, torch.ones(4))


@pytest.mark.parametrize(
    "kwargs", [{"num_samples": 1}, {"num_samples": 2.5}, {"num_samples": True}, {"correction": 2}]
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        mc_dropout_predict(nn.Dropout(), torch.ones(4), **kwargs)


@pytest.mark.parametrize("bad", [lambda x: (x,), lambda x: x.long(), lambda x: x * float("nan")])
def test_bad_outputs_restore_flags(bad):
    model = nn.Dropout().eval()
    with pytest.raises((ValueError, TypeError)):
        mc_dropout_predict(model, torch.ones(4), output_transform=bad)
    assert not model.training


def test_changing_shape_rejected():
    outputs = iter([torch.ones(2), torch.ones(3)])
    with pytest.raises(ValueError, match="remain constant"):
        mc_dropout_predict(nn.Dropout(), None, predict_fn=lambda *_: next(outputs))


def test_forward_exception_disables_gradients_and_restores_modes():
    model = nn.Dropout().eval()

    def fail(net, inputs):
        assert not torch.is_grad_enabled()
        assert net.training
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        mc_dropout_predict(model, torch.ones(4, requires_grad=True), predict_fn=fail)
    assert not model.training


def test_class_selection_preserves_batch_and_spatial_axes():
    std = torch.arange(24).float().reshape(2, 3, 4)
    classes = torch.tensor([[0, 1, 2, 0], [2, 1, 0, 2]])
    actual = select_class_uncertainty(std, classes)
    expected = torch.stack([std[b, classes[b], torch.arange(4)] for b in range(2)])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        select_class_uncertainty(std.movedim(1, -1), classes, class_dim=-1),
        expected,
    )
    with pytest.raises(ValueError):
        select_class_uncertainty(std, classes + 3)
    with pytest.raises(ValueError):
        select_class_uncertainty(std, classes.float())
