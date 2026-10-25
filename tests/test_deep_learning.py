import json
import pytest
import torch
from deep_learning import train, load_model, benchmark, export, device_for


def test_cpu_lifecycle(tmp_path):
    torch.set_num_threads(1)
    metrics = train(tmp_path, 'cpu', epochs=6)
    assert metrics['holdout_accuracy'] > 0.85
    assert metrics['loss'][-1] < metrics['loss'][0]
    assert load_model(tmp_path)(torch.zeros(3, 16)).shape == (3, 2)
    assert benchmark(tmp_path, 'cpu', batch=8, iterations=3)['parity_passed']
    export(tmp_path)
    import onnxruntime as ort
    import numpy as np
    session = ort.InferenceSession(str(tmp_path / 'model.onnx'), providers=['CPUExecutionProvider'])
    for batch in [1, 7]:
        x = torch.randn(batch, 16)
        with torch.inference_mode():
            expected = load_model(tmp_path)(x).numpy()
        np.testing.assert_allclose(session.run(None, {'features': x.numpy()})[0], expected, rtol=1e-4, atol=1e-5)


def test_invalid_configuration(tmp_path):
    with pytest.raises(ValueError):
        train(tmp_path, 'cpu', epochs=0)
    with pytest.raises(ValueError):
        train(tmp_path, 'cpu', amp=True)
    with pytest.raises(ValueError):
        benchmark(tmp_path, batch=0)
    if not torch.cuda.is_available():
        with pytest.raises(ValueError):
            device_for('cuda')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires NVIDIA GPU')
def test_cuda_amp(tmp_path):
    metrics = train(tmp_path, 'cuda', epochs=6, amp=True)
    assert metrics['amp'] and metrics['holdout_accuracy'] > 0.85
    assert benchmark(tmp_path, 'cuda', iterations=3)['parity_passed']
