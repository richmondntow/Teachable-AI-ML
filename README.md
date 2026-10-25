# Teachable-AI-ML
Teachable AI/ML is a beginner-friendly project that helps students and learners understand the basics of Artificial Intelligence (AI) and Machine Learning (ML). With a simple HTML, CSS, and JavaScript frontend, users can upload data, train simple models, and visualize how AI learns step by step—without needing complex tools or frameworks.
🎓 Educational Goals
Learn how datasets are used to train AI.
Understand the difference between classification, regression, and clustering.
See training progress with real-time visualizations.
Experiment with different parameters to compare results.
Export models for reuse in projects.
🚀 Features
📂 Upload Data: Load CSV files or images into the platform.
⚡ No Frameworks Needed: Built entirely with HTML, CSS, and JavaScript.
📊 Visual Training: Watch accuracy and loss values change as the model learns.
🤖 Prebuilt Models: Train simple models like decision trees or linear regression.
🧩 Interactive Controls: Adjust training settings (learning rate, epochs, etc.).
💾 Export Models: Save trained models for future use.
🛠️ Tech Stack
Frontend: HTML, CSS, JavaScript
ML Libraries: TensorFlow.js (for in-browser training)
Visualization: Chart.js / D3.js
📦 Installation & Setup
Clone the repository
git clone https://github.com/richmondntow/teachable-ai-ml.git
cd teachable-ai-ml
Open the project
Since this is a web-based project, simply open index.html in your browser.
open index.html   # Mac
start index.html  # Windows
(If you prefer running a local server, use Live Server extension in VS Code.)
🎯 How to Use
Open the app in your browser.
Upload a dataset (CSV or images).
Select the type of model you want to train.
Adjust settings like epochs or learning rate.
Click Train Model to start learning.
Watch graphs update in real time.
Export your trained model if you want to reuse it.
📊 Example Learning Scenarios
Classify handwritten numbers (MNIST).
Predict housing prices with CSV data.
Experiment with small image datasets (cats vs dogs).
Understand overfitting vs underfitting by changing epochs.
🤝 Contributing
This project is designed to help students learn AI/ML. If you’d like to contribute:
Fork the repo
Create a new branch (feature-new-idea)
Commit changes
Open a pull request
📜 License
free to use, modify, and learn from.

## PyTorch CUDA training and TensorRT inference lab

The original scikit-learn web app remains available. `deep_learning.py` adds an
optional CLI lab with a three-layer neural network on a **synthetic** 16-feature
classification dataset. This is a demonstrator, not a production training service.
No GPU speedup or production accuracy claim is implied.

### CPU installation and validation

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-deep-learning.txt
python -m pytest tests/test_deep_learning.py -q
python deep_learning.py train --device cpu
python deep_learning.py benchmark --device cpu
python deep_learning.py export
```

### NVIDIA GPU training

Install a CUDA-compatible PyTorch wheel using the official PyTorch installer for
your driver: https://pytorch.org/get-started/locally/ . Then run:

```sh
python deep_learning.py train --device cuda --amp --output artifacts/cuda-amp
python deep_learning.py benchmark --device cuda --output artifacts/cuda-amp
python deep_learning.py export --output artifacts/cuda-amp
```

The training loop moves model and batches to CUDA and uses autocast plus gradient
scaling for optional FP16 mixed precision. Explicit CUDA requests fail clearly
when unavailable; `--device auto` falls back to CPU. JSON reports record hardware,
PyTorch/CUDA versions, seed, loss, holdout accuracy and elapsed training time.
Model checkpoints contain only state dictionaries and metadata.

### Optional TensorRT inference

TensorRT accelerates inference, not this training loop. In a separate NVIDIA GPU
environment install compatible CUDA, cuDNN, TensorRT and `onnxruntime-gpu` per
https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html .
Remove CPU `onnxruntime` before installing the GPU package. Then:

```sh
python deep_learning.py benchmark --backend tensorrt --device cuda --output artifacts/cuda-amp
```

This explicitly selects the TensorRT execution provider, enables FP16 and engine
caching, verifies numerical and prediction parity against PyTorch CPU, warms up,
and writes measured throughput/latency. CUDA fallback is allowed for unsupported
operators: this is not proof of complete TensorRT graph coverage. Inspect ONNX
Runtime profiling before claiming all layers run in TensorRT. Engines are specific
to hardware/software combinations and should be rebuilt when the environment changes.

### Honest performance comparisons

Use the same checkpoint, batch size, iterations and machine for comparisons.
PyTorch CUDA timing synchronizes before and after measurement. Both backends
start with CPU inputs and return CPU outputs, including host/device transfers
inside their timed loops. Compare end-to-end latency, not isolated kernel speed. The tiny network may be faster on CPU. Do not publish
speedup figures until measured on real hardware. CPU CI checks learning, checkpoint
reload, dynamic-batch ONNX parity, inference timing and invalid configurations;
CUDA tests skip without hardware. GPU and TensorRT paths require separate validation.
