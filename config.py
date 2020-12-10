from collections import OrderedDict
import utils

# Files and extensions
DATA_DIR = '/content/drive/MyDrive/dlrs-data/dlrs-data/'
TEXT_DIR = DATA_DIR + 'text/'
AUDIO_DIR = DATA_DIR + 'audio/'
AUDIOFILES_DIR = AUDIO_DIR + 'audio_files/'
SPECTRO_DIR = AUDIO_DIR + 'spectro/'
PATCHES_DIR = AUDIO_DIR + 'patches/'
TRAINDATA_DIR = DATA_DIR + 'train_data/'
TESTDATA_DIR = DATA_DIR + 'test_data/'
SPLITS_DIR = DATA_DIR + 'splits/'
MODELS_DIR = 'models/'
DEFAULT_TRAINED_MODELS_FILE = MODELS_DIR + 'trained_models.tsv'
DEFAULT_MODEL_PREFIX = 'model_'
RESULTS_DIR = 'results/'
PREDICTIONS_DIR = RESULTS_DIR + 'predictions/'
MODEL_EXT = '.json'
WEIGHTS_EXT = '.h5'
MAX_N_SCALER = 300000

# Configs for spectrograms and patches
CONFIG_AUDIO = {
    'dataset': 'dummy',
    'index_file' : 'index_audio_dummy.tsv',
    'num_process' : 8,
    'resample_sr' : 22050,
    'spectrogram_type' : 'cqt',
    'hop' : 1024,
    'cqt_bins' : 96,
    'n_fft': 512,
    'n_mels': 96,
    'audio_ext' : ['mp3'],
    'n_samples': 1,
    'seconds': 15,
}

# Parameters
PARAMS = {
    # dataset params
    'dataset': {
        'fact': 'als',
        'dim': 200,
        'dataset_ab': 'MSD-A-artists',
        'dataset_as': 'MSD-A-songs',
        'window': 15,
        'nsamples': 'all',
        'npatches': 1,
        'evaluation': 'recommendation',
        'configuration': 'rec_multi',
        'meta-suffix': 'sem-bio',
        'meta-suffix2': 'audio-emb',
        'num_users': 10000,
    },
    # training params
    'training' : {
        'decay': 1e-6,
        'learning_rate': 0.1,
        'momentum': 0.95,
        'n_epochs': 1,
        'n_minibatch': 256,
        'nesterov': True,
        'validation': 0.1,
        'test': 0.1,
        'loss_func': 'cosine',
        'val_from_file': False,
        'normalize_y': True
    },
    # cnn params
    'cnn' : {
        'n_ab_features': 2048,
        'n_dense_ab' : 512,
        'n_frames' : 322,
        'n_mel' : 96,
        'n_filters_1' : 256,
        'n_filters_2' : 512,
        'n_filters_3' : 1024,
        'n_filters_4' : 1024,
        'n_kernel_1' : (4, 96),
        'n_kernel_2' : (4, 1),
        'n_kernel_3' : (4, 1),
        'n_kernel_4' : (1, 1),
        'n_pool_1' : (4, 1),
        'n_pool_2' : (4, 1),
        'n_pool_3' : (1, 1),
        'n_pool_4' : (1, 1),
        'n_dense_concat' : 512,
        'n_out' : 200,
        'dropout_factor_ab' : 0.5,
        'dropout_factor_cnn' : 0.5,
        'dropout_factor_concat': 0.7,
        'final_activation': 'linear',
    },
    'predicting' : {
        'trim_coeff' : 0.15
    },
    'evaluating' : {
        'get_map' : False,
        'get_p' : True,
    }
}


class Config(object):
    """Configuration for the training process."""
    def __init__(self, params, normalize=False, whiten=True):
        self.model_id = utils.get_next_model_id()
        self.norm = normalize
        self.whiten = whiten
        self.dataset_settings = params['dataset']
        self.training_params = params['training']
        self.model_arch = params['cnn']
        self.predicting_params = params['predicting']

    def get_dict(self):
        object_dict = self.__dict__
        first_key = "model_id"
        conf_dict = OrderedDict({first_key: object_dict[first_key]})
        conf_dict.update(object_dict)
        return conf_dict
