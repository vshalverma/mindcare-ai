import torch
import transformers
import streamlit
import pandas as pd
import numpy as np
import sklearn

print("=" * 50)
print("✅  SETUP VERIFICATION")
print("=" * 50)
print(f"Python libraries installed:")
print(f"  PyTorch:        {torch.__version__}")
print(f"  Transformers:   {transformers.__version__}")
print(f"  Streamlit:      {streamlit.__version__}")
print(f"  Pandas:         {pd.__version__}")
print(f"  NumPy:          {np.__version__}")
print(f"  Scikit-learn:   {sklearn.__version__}")
print("=" * 50)
print(f"🖥️  GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
      print(f"   GPU Name:      {torch.cuda.get_device_name(0)}")
      print(f"   GPU Memory:    {torch.cuda.get_device_properties(0).total_memory/ 1e9:.2f} GB")
print("=" * 50)
print("🎉 ALL READY! Setup complete.")
