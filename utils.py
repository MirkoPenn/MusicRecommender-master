import pandas as pd
import glob, json, os
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from config import MODELS_DIR, DEFAULT_MODEL_PREFIX, MODEL_EXT

def ensure_dir(directory):
    """Makes sure that the given directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_next_model_id(models_dir=MODELS_DIR,
                      model_prefix=DEFAULT_MODEL_PREFIX,
                      model_ext=MODEL_EXT):
    """Gets the next model id based on the models in `models_dir`
    directory."""
    models = glob.glob(os.path.join(models_dir, model_prefix + '*'))
    return model_prefix + str(len(models) + 1)

def load_model(model_file):
    """Load the model into the given model file."""
    with open(model_file, 'w') as f:
        json_string = json.load(f)
    model = model_from_json(json_string)

    return model

def save_model(model, model_file):
    """Save the model into the given model file."""
    json_string = model.to_json()
    with open(model_file, 'w') as f:
        json.dump(json_string, f)

def save_trained_model(trained_models_file, trained_model):
    try:
        df = pd.read_csv(trained_models_file, sep='\t')
    except IOError:
        df = pd.DataFrame(columns=trained_model.keys())
    df = df.append(trained_model, ignore_index=True)
    df.to_csv(trained_models_file, sep='\t', index=False)
