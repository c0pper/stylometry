from pathlib import Path
import pandas as pd
import inspect

from stylometry_utils.class_Experiment import Experiment
e = Experiment(Path("../fn20k.xlsx"), "label")
ds = Path(r"C:\Users\smarotta\PycharmProjects\stylometry\fn20k.xlsx").stem
colab_path = r"C:\Users\smarotta\PycharmProjects\stylometry\logs"
path = Path(Path(colab_path) / ds)
path.mkdir(parents=True, exist_ok=True)
print(path)

