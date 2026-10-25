"""Reproducible tabular deep-learning demo: train, export, benchmark."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch import nn


def device_for(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; install a CUDA-enabled PyTorch build on an NVIDIA host.')
    return torch.device('cuda' if name == 'auto' and torch.cuda.is_available() else ('cpu' if name == 'auto' else name))


def network():
    return nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))


def dataset(seed, count):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(count, 16, generator=g)
    y = (x[:, :4].sum(dim=1) > 0).long()
    return x, y


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def train(output, device='auto', epochs=10, seed=42, amp=False):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    target = device_for(device)
    if amp and target.type != 'cuda':
        raise ValueError('Mixed precision requires CUDA in this demo; omit --amp on CPU.')
    torch.manual_seed(seed)
    model = network().to(target)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    # Separate generators create independent training and holdout samples.
    x, y = dataset(seed, 2048)
    test_x, test_y = dataset(seed + 1, 512)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=128,
                                        shuffle=True, generator=torch.Generator().manual_seed(seed),
                                        pin_memory=target.type == 'cuda')
    sync(target)
    start = time.perf_counter()
    history = []
    for _ in range(epochs):
        model.train()
        loss_total = 0.0
        for features, labels in loader:
            features = features.to(target, non_blocking=True)
            labels = labels.to(target, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=target.type, dtype=torch.float16, enabled=amp):
                loss = nn.functional.cross_entropy(model(features), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_total += loss.detach().item() * len(labels)
        history.append(loss_total / len(x))
    sync(target)
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.inference_mode():
        accuracy = (model(test_x.to(target)).argmax(1).cpu() == test_y).float().mean().item()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({ 'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'seed': seed}, output / 'model.pt')
    metrics = dict(device=str(target), gpu=torch.cuda.get_device_name(target) if target.type == 'cuda' else None,
                   torch_version=torch.__version__, cuda_version=torch.version.cuda, amp=amp, seed=seed,
                   dataset='synthetic 16-feature binary classification', epochs=epochs, training_samples=len(x),
                   holdout_samples=len(test_x), holdout_accuracy=accuracy, training_seconds=elapsed, loss=history)
    (output / 'training.json').write_text(json.dumps(metrics, indent=2))
    return metrics


def load_model(directory):
    checkpoint = torch.load(Path(directory) / 'model.pt', map_location='cpu', weights_only=True)
    model = network()
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


def export(directory):
    # Legacy exporter avoids an additional onnxscript dependency for this small network.
    import onnx
    model = load_model(directory)
    path = Path(directory) / 'model.onnx'
    torch.onnx.export(model, torch.zeros(1, 16), str(path), input_names=['features'], output_names=['logits'],
                      dynamic_axes={'features': {0: 'batch'}, 'logits': {0: 'batch'}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    return str(path)


def benchmark(directory, device='cpu', backend='torch', batch=128, iterations=100):
    if batch < 1 or iterations < 1:
        raise ValueError('batch and iterations must be positive')
    target = device_for(device)
    model = load_model(directory)
    x, _ = dataset(123, batch)
    with torch.inference_mode():
        expected = model(x)
    if backend == 'tensorrt':
        if target.type != 'cuda':
            raise ValueError('TensorRT requires --device cuda')
        import onnxruntime as ort
        if 'TensorrtExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('Install compatible ONNX Runtime GPU, CUDA, cuDNN and TensorRT libraries.')
        session = ort.InferenceSession(str(Path(directory) / 'model.onnx'), providers=[
            ('TensorrtExecutionProvider', {'trt_engine_cache_enable': True,
             'trt_engine_cache_path': str(Path(directory) / 'trt-cache'), 'trt_fp16_enable': True}),
            'CUDAExecutionProvider'])
        def run():
            return torch.from_numpy(session.run(['logits'], {'features': x.numpy()})[0])
    else:
        model = model.to(target)
        def run():
            # Match ORT's CPU input/output boundary for an end-to-end comparison.
            return model(x.to(target)).cpu()
    with torch.inference_mode():
        # Accuracy gate before timing; FP16 permits small numerical differences.
        actual = run().cpu()
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        if not torch.equal(actual.argmax(1), expected.argmax(1)):
            raise RuntimeError('Prediction parity failed')
        for _ in range(10):
            run()
        sync(target)
        start = time.perf_counter()
        for _ in range(iterations):
            run()
        sync(target)
        elapsed = time.perf_counter() - start
    report = dict(backend=backend, device=str(target), batch=batch, iterations=iterations,
                  mean_batch_ms=1000 * elapsed / iterations, samples_per_second=batch * iterations / elapsed,
                  parity_passed=True, includes_host_transfers=target.type == 'cuda',
                  note='TensorRT provider can partition unsupported operations to CUDA; inspect ORT profiling for coverage.')
    (Path(directory) / f'benchmark-{backend}-{target.type}.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['train', 'export', 'benchmark'])
    parser.add_argument('--output', default='artifacts/deep-learning')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--backend', choices=['torch', 'tensorrt'], default='torch')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    try:
        if args.action == 'train':
            result = train(args.output, args.device, args.epochs, args.seed, args.amp)
        elif args.action == 'export':
            result = export(args.output)
        else:
            result = benchmark(args.output, args.device, args.backend, args.batch, args.iterations)
        print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f'{error}\n')
