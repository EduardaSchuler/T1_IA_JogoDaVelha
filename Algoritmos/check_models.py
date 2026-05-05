import sys
from pathlib import Path
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from frontend import MODEL_REGISTRY
print('MODEL KEYS', list(MODEL_REGISTRY.keys()))
for k, m in MODEL_REGISTRY.items():
    print('---', k, m.name)
    print('model file:', m.model_file)
    print('available:', m.is_available())
    if m.is_available():
        try:
            print('predict:', m.predict(['b']*9))
        except Exception as e:
            print('ERROR:', type(e).__name__, e)
