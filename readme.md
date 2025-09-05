Ohayou :🕊️: ZFG



## Environment setup
```
uv venv --python 3.12 --seed .venv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install ipykernel
python -m ipykernel install --user --name=gemma_3_avt_zfg --display-name "Python (gemma_3_avt_zfg)"
uv pip install -U transformers
uv pip install accelerate
uv pip install nvitop
```
