# Installation Guide

## Step 1: Install Basic Requirements

First, install the core requirements without TensorFlow:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib scipy deap sympy
```

## Step 2: Install TensorFlow (Optional)

TensorFlow is optional but recommended for the best ANN performance. Choose one option based on your system:

### Option A: Standard CPU Version (Most Common)

```bash
pip install tensorflow
```

### Option B: For Apple Silicon Macs (M1/M2)

```bash
pip install tensorflow-macos
# Optional GPU acceleration:
pip install tensorflow-metal
```

### Option C: For Systems with NVIDIA GPU

```bash
pip install tensorflow-gpu
```

### Option D: Lightweight CPU-only Version

```bash
pip install tensorflow-cpu
```

## If TensorFlow Installation Fails

Don't worry! The system will automatically use scikit-learn's MLPRegressor as a fallback for the ANN model. You can still use all other models (Linear Regression, Random Forest, Gradient Boosting, SVR, Hybrid, Bagging, and GEP).

## Troubleshooting

### Common Issues:

1. **Python Version**: Make sure you're using Python 3.7 or higher
2. **Virtual Environment**: Consider using a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **System-specific TensorFlow Issues**:

   - **Windows**: Try `pip install tensorflow-cpu` first
   - **macOS**: If you have an M1/M2 Mac, use `tensorflow-macos`
   - **Linux**: Standard `tensorflow` should work

4. **If all else fails**, you can run without TensorFlow:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn joblib scipy deap sympy
   ```
   The system will automatically detect that TensorFlow is not available and use the scikit-learn ANN implementation instead.

## Verification

To verify your installation, run:

```python
python -c "import numpy, pandas, sklearn, matplotlib, seaborn, scipy, deap, sympy; print('All basic packages installed successfully!')"
```

To check if TensorFlow is available:

```python
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} is available")
except ImportError:
    print("TensorFlow not available - will use scikit-learn ANN fallback")
```

## Quick Start

Once installed, you can start training models:

```bash
# Train all models (this will automatically detect available packages)
python main_enhanced.py --model all

# Train just the models that don't require TensorFlow
python main_enhanced.py --model bagging
python main_enhanced.py --model gep
```
