@echo off
echo Installing dependencies for PID training...
python -m pip install --upgrade pip
pip install torch --index-url https://download.python.org/whl/cu118
pip install peft transformers pandas numpy scikit-learn joblib onnx onnxruntime-gpu
pip install pymavlink dronekit
echo Done!
pause