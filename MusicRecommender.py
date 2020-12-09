import keras
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Lambda, Input, concatenate
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.regularizers import l2
from keras.models import Model, model_from_json
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import theano
import theano.tensor as tt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import r2_score
from scipy.sparse import csr_matrix

import pandas as pd
import numpy as np
import argparse, copy, json, time, os, gc, shutil, tempfile, glob
import signal, librosa, pickle, h5py, traceback
from os import environ
from importlib import reload
from joblib import Parallel, delayed

import utils
from config import *


class MusicRecommender:
    '''This is Music Reommender class using MSD dataset.'''

    def __init__(self, preprocess=False):
        # Change keras backend
        def set_keras_backend(backend):
            if K.backend() != backend:
                environ['KERAS_BACKEND'] = 'theano'
            reload(K)
            assert K.backend() == backend
        set_keras_backend('theano')

        # Set config
        self.__config = Config(PARAMS)

        # Preprocess data
        if preprocess:
            self.preprocess_data(
                PARAMS['dataset']['dataset_ab'], PARAMS['dataset']['dataset_as'],
                PARAMS['dataset']['meta-suffix'], PARAMS['dataset']['meta-suffix2']
            )
        print('Preprocessing is completed.\n')

    def load_sparse_csr(self, filename):
        '''Load spare csr matrix from a file.'''

        loader = np.load(filename)
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

    def save_sparse_csr(self, filename, array):
        '''Save spare csr matrix to a file.'''

        np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)

    def calc_spectogram(self, id, audio_file, spectro_file, dataset_name, set_name):
        '''Calculate spectograms for audio files'''

        def signal_handler(signum, frame):
            raise Exception("Timed out!")

        try:
            if not os.path.exists(spectro_file[:spectro_file.rfind('/')+1]):
                os.makedirs(spectro_file[:spectro_file.rfind('/')+1])

            if not os.path.isfile(spectro_file):
                signal.signal(signal.SIGALRM, signal_handler)
                signal.alarm(50)

                # Get actual audio and calculate spectrogram
                audio, sr = librosa.load(audio_file, sr=CONFIG_AUDIO['resample_sr'])
                if CONFIG_AUDIO['spectrogram_type'] == 'cqt':
                    spec = librosa.cqt(audio, sr=sr, hop_length=CONFIG_AUDIO['hop'], n_bins=CONFIG_AUDIO['cqt_bins'])
                elif CONFIG_AUDIO['spectrogram_type'] == 'mel':
                    spec = librosa.feature.melspectrogram(
                        y=audio, sr=sr, hop_length=CONFIG_AUDIO['hop'],
                        n_fft=CONFIG_AUDIO['n_fft'], n_mels=CONFIG_AUDIO['n_mels']
                    )
                elif CONFIG_AUDIO['spectrogram_type'] == 'stft':
                    spec = librosa.stft(y=audio, n_fft=CONFIG_AUDIO['n_fft'])

                # Write results
                with open(spectro_file, 'wb') as f:
                    pickle.dump(spec, f, protocol=-1)

            # Append spectrogram index
            fw = open(SPECTRO_DIR + 'index_spectro_%s_%s.tsv' % (set_name, dataset_name), 'a')
            fw.write("%s\t%s\t%s\n" % (id, spectro_file[len(SPECTRO_DIR):], audio_file))
            fw.close()

            print('Computed spec: %s' % audio_file)
        except Exception:
            print('Error computing spec', audio_file)
            print(traceback.format_exc())

    def apply_patches(self, dataset_name, set_name, data_dir, scaler=None):
        '''Apply patch concept to spectograms.'''

        def sample_patch(spec, n_frames):
            '''Randomly sample a part of the mel spectrogram.'''

            if n_frames <= spec.shape[0]:
                r_idx = np.random.randint(0, high=spec.shape[0] - n_frames + 1)
                return spec[r_idx:r_idx + n_frames]
            else:
                return np.vstack((spec, np.zeros((n_frames - spec.shape[0], spec.shape[1]))))

        def scale(X, scaler=None, max_N=MAX_N_SCALER):
            '''Scale numpy array by StandardScaler.'''

            shape = X.shape
            X.shape = (shape[0], shape[2] * shape[3])
            if not scaler:
                scaler = StandardScaler()
                N = min([len(X), max_N])
                scaler.fit(X[:N])
            X = scaler.transform(X)
            X.shape = shape
            return X, scaler

        # Get a file and items
        if not os.path.exists(PATCHES_DIR): os.makedirs(PATCHES_DIR)
        f = h5py.File(
            PATCHES_DIR + '/patches_%s_%s_%sx%s_tmp.hdf5'
            % (set_name, dataset_name, CONFIG_AUDIO['n_samples'], CONFIG_AUDIO['seconds']), 'w'
        )
        items = open(data_dir + '/items_index_%s_%s' % (set_name, dataset_name) + '.tsv').read().splitlines()
        n_items = len(items) * CONFIG_AUDIO['n_samples']
        print('\nThe number audio items:', n_items)

        # Create features and index
        N_FRAMES = int(CONFIG_AUDIO['seconds'] * CONFIG_AUDIO['resample_sr'] / float(CONFIG_AUDIO['hop']))
        x_dset = f.create_dataset('features', (n_items, 1, N_FRAMES, CONFIG_AUDIO['cqt_bins']), dtype='f')
        i_dset = f.create_dataset('index', (n_items,), maxshape=(n_items,), dtype='S2')

        # Get items and item indices
        k = 0
        itemset = []
        itemset_index = []
        for t, tid in enumerate(items):
            file = SPECTRO_DIR + set_name + '/' + tid + '.pk'
            spec = pickle.load(open(file, 'rb'))
            spec = librosa.power_to_db(np.abs(spec)**2, ref=np.max).T
            for i in range(0, CONFIG_AUDIO['n_samples']):
                sample = sample_patch(spec, N_FRAMES)
                x_dset[k, :, :, :] = sample.reshape(-1, sample.shape[0], sample.shape[1])
                i_dset[k] = tid.encode('ascii')
                itemset.append(tid)
                itemset_index.append(t)
                k += 1
                
        # Clean empty spectrograms
        print('Cleaning empty spectrograms.')
        f2 = h5py.File(
            PATCHES_DIR + '/patches_%s_%s_%sx%s.hdf5'
            % (set_name, dataset_name, CONFIG_AUDIO['n_samples'], CONFIG_AUDIO['seconds']),
            'w'
        )
        index = f['index'][:]
        index_clean = np.where(index != b'')[0]
        n_items = len(index_clean)
        x_dset2 = f2.create_dataset('features', (n_items, 1, N_FRAMES, CONFIG_AUDIO['cqt_bins']), dtype='f')
        i_dset2 = f2.create_dataset('index', (n_items,), maxshape=(n_items,), dtype='S2')
        for i in range(0, n_items):
            x_dset2[i] = x_dset[index_clean[i]]
            i_dset2[i] = i_dset[index_clean[i]]

        f.close()
        os.remove(
            PATCHES_DIR + 'patches_%s_%s_%sx%s_tmp.hdf5'
            % (set_name, dataset_name, CONFIG_AUDIO['n_samples'], CONFIG_AUDIO['seconds'])
        )

        # Normalize
        print('Normalizing...')
        block_step = 10000
        for i in range(0, len(itemset), block_step):
            x_block = x_dset2[i:min(len(itemset), i+block_step)]
            x_norm, scaler = scale(x_block, scaler)
            x_dset2[i:min(len(itemset), i+block_step)] = x_norm
        scaler_file = PATCHES_DIR + '/scaler_%s_%s_%sx%s.pk' \
                      % (set_name, dataset_name, CONFIG_AUDIO['n_samples'], CONFIG_AUDIO['seconds'])
        pickle.dump(scaler, open(scaler_file, 'wb'))

        return x_dset2, scaler

    def preprocess_audio_data(self, dataset, set_name, data_dir):
        '''Preprocess audio data.'''

        # create empty spectrograms index
        if not os.path.exists(SPECTRO_DIR): os.makedirs(SPECTRO_DIR)
        if not os.path.exists(SPECTRO_DIR + set_name): os.makedirs(SPECTRO_DIR + set_name)
        fw = open(SPECTRO_DIR + 'index_spectro_%s_%s.tsv' % (set_name, dataset), 'w')
        fw.close()

        # list audios to process: according to 'index_file'
        if not os.path.exists(AUDIO_DIR): os.makedirs(AUDIO_DIR)
        files_to_convert = []
        f_index = open(AUDIO_DIR + CONFIG_AUDIO['index_file'])
        f_items_index = open(data_dir + '/items_index_%s_%s.tsv' % (set_name, dataset), 'w')
        for line in f_index.readlines():
            id, audio = line.strip().split('\t')
            f_items_index.write(id + '\n')
            spect = id + '.pk'
            files_to_convert.append((id, AUDIOFILES_DIR + audio, SPECTRO_DIR + set_name + '/' + spect))
        f_items_index.close()
        print(len(files_to_convert), 'audio files to process!')

        # compute spectrogram
        Parallel(n_jobs=CONFIG_AUDIO['num_process'])(
            delayed(self.calc_spectogram)(id, audio_file, spectro_file, dataset, set_name)
            for id, audio_file, spectro_file in files_to_convert
        )

        # save parameters
        json.dump(CONFIG_AUDIO, open(SPECTRO_DIR + 'params_%s.json' % set_name, 'w'))

        # Apply patches
        track_embeddings, _ = self.apply_patches(dataset, set_name, data_dir)
        X_file = data_dir + '/X_%s_%s' % (set_name, dataset)
        np.save(X_file, track_embeddings)

        print('\nAudio preprocessing is completed.')

    def preprocess_data(self, dataset_ab, dataset_as, meta_source_ab,
                        meta_source_as, biography_data=False, audio_data=False):
        '''Preprocess artist biographies and audio files.'''

        if biography_data:
            train = open(TRAINDATA_DIR + '/items_index_train_%s.tsv' % dataset_ab).read().splitlines()
            test = open(TESTDATA_DIR + '/items_index_test_%s.tsv' % dataset_ab).read().splitlines()

            N_WORDS = 300
            texts = dict()
            files = glob.glob(TEXT_DIR + '/*.txt')
            for file in files:
                id = file[file.rfind('/')+1:file.rfind('.')]
                text = open(file).read()
                sentences = text.split('\n')
                clean_sentences = [s.split(' ') for s in sentences]
                words = [word for s in clean_sentences for word in s]
                texts[id] = ' '.join(words[:N_WORDS])

            vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', max_features=10000)
            X_train = vectorizer.fit_transform([texts[item] for item in train])
            X_test = vectorizer.transform([texts[item] for item in test])
            print(X_train.shape, X_test.shape)

            if not os.path.isdir(TRAINDATA_DIR): os.makedirs(TRAINDATA_DIR)
            if not os.path.isdir(TESTDATA_DIR): os.makedirs(TESTDATA_DIR)

            X_file = TRAINDATA_DIR + '/X_train_%s_%s' % (meta_source_ab, dataset_ab)
            np.savez(X_file, data=X_train.data, indices=X_train.indices, indptr=X_train.indptr, shape=X_train.shape)
            X_file = TESTDATA_DIR + '/X_test_%s_%s' % (meta_source_ab, dataset_ab)
            np.savez(X_file, data=X_test.data, indices=X_test.indices, indptr=X_test.indptr, shape=X_test.shape)

        if audio_data:
            self.preprocess_audio_data(CONFIG_AUDIO['dataset'], 'train', TRAINDATA_DIR)
            # self.preprocess_audio_data(CONFIG_AUDIO['dataset'], 'test', TESTDATA_DIR)
    
    def load_data(self, params, dataset_ab, dataset_as, val_percent,
                  test_percent, n_samples, meta_source_ab, meta_source_as):
        '''Load data for training, validation and test before testing using whole data.'''

        # Get metadata for X and Y
        all_X_meta_ab = self.load_sparse_csr(TRAINDATA_DIR + 'X_train_%s_%s.npz' % (meta_source_ab, dataset_ab))
        PARAMS['cnn']['n_metafeatures'] = all_X_meta_ab.shape[1]
        all_X_meta_as = np.load(TRAINDATA_DIR + 'X_train_%s_%s.npy' % (meta_source_as, dataset_as))
        PARAMS['cnn']['n_metafeatures2'] = len(all_X_meta_as[0])
        all_Y = np.load(SPLITS_DIR + 'y_train_als_200_MSD-A-songs.npy')
        normalize(all_Y, copy=False)

        # Calculate the number of sample for train and validation
        N = all_Y.shape[0]
        train_percent = 1 - val_percent - test_percent
        N_train = int(train_percent * N)
        N_val = int(val_percent * N)

        # Get train, val, test data
        X_train = [all_X_meta_ab[:N_train], all_X_meta_as[:N_train]]
        X_val = [all_X_meta_ab[N_train:N_train + N_val], all_X_meta_as[N_train:N_train + N_val]]
        X_test = [all_X_meta_ab[N_train + N_val:], all_X_meta_as[N_train + N_val:]]
        Y_train = all_Y[:N_train]
        Y_val = all_Y[N_train:N_train + N_val]
        Y_test = all_Y[N_train + N_val:]

        print("Training data points: %d" % N_train)
        print("Validation data points: %d" % N_val)
        print("Test data points for: %d" % (N - N_train - N_val), '\n')

        return X_train, Y_train, X_val, Y_val, X_test, Y_test

    def build_model(self, config, audio_data=False):
        '''Build music recommendation model based on CNN and FF.'''

        def cosine(x, y):
            squared_magnitude = lambda x: tt.sqr(x).sum(axis=-1)
            magnitude = lambda x: tt.sqrt(tt.maximum(squared_magnitude(x), np.finfo(x.dtype).tiny))
            return tt.clip((1 - (x * y).sum(axis=-1) / (magnitude(x) * magnitude(y))) / 2, 0, 1)

        # Get params
        params = config.model_arch

        # Input layer for audio biography
        inputs_ab = Input(shape=(params["n_metafeatures"],))
        input_layer_ab = Dropout(params["dropout_factor_ab"])(inputs_ab)

        dense_layer_ab = Dense(output_dim=params["n_dense_ab"], init='uniform', activation='relu')(input_layer_ab)
        dense_layer_ab = Dropout(params["dropout_factor_ab"])(dense_layer_ab)
        dense_layer_ab = Dense(output_dim=params["n_dense_ab"], init='uniform', activation='relu')(dense_layer_ab)
        dense_layer_ab = Dropout(params["dropout_factor_ab"])(dense_layer_ab)
        dense_layer_ab = Lambda(lambda x: K.l2_normalize(x, axis=1))(dense_layer_ab)

        audio_embedding_layer = None
        if audio_data:
            # Input layer for audio
            inputs_as = Input(shape=(1, params['n_frames'], params['n_mel']))

            conv_layer = inputs_as
            for i in range(4):
                conv_layer = Convolution2D(
                    params['n_filters_' + str(i+1)], params['n_kernel_' + str(i+1)][0], params['n_kernel_' + str(i+1)][1],
                    border_mode='valid', activation='relu', init='uniform'#,
                    # input_shape=(1, params["n_frames"], params["n_mel"])
                )(conv_layer)
                conv_layer = MaxPooling2D(pool_size=(params['n_pool_' + str(i+1)][0], params['n_pool_'  + str(i+1)][1]))(conv_layer)
                conv_layer = Dropout(params['dropout_factor_cnn'])(conv_layer)
            conv_layer = Flatten()(conv_layer)
            audio_embedding_layer = conv_layer
        else:
            # Input layers for audio embedding layer
            inputs2 = Input(shape=(params['n_metafeatures2'],))
            input_layer2 = Lambda(lambda x: K.l2_normalize(x, axis=1))(inputs2)
            audio_embedding_layer = input_layer2

        # Concatenate layer
        concat_layer = concatenate([dense_layer_ab, audio_embedding_layer], axis=1)
        concat_layer = Dropout(params["dropout_factor_concat"])(concat_layer)
        concat_layer = Dense(output_dim=params["n_dense_concat"], init="uniform", activation='relu')(concat_layer)

        # Output layer
        output_layer = Dense(output_dim=params["n_out"], init="uniform", activation=params['final_activation'])(concat_layer)
        output_layer = Lambda(lambda x: K.l2_normalize(x, axis=1))(output_layer)

        # Final model
        model = Model(input=[inputs_ab, inputs2], output=output_layer)

        # Learning setups
        optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        metrics = ['mean_squared_error']
        model.compile(loss=cosine, optimizer=optimizer, metrics=metrics)

        return model

    def recommend(self):
        '''Recommend track IDs using artist biographies and audio files.'''

        # Load data
        print('Loading Data...')
        X_train, Y_train, X_val, Y_val, X_test, Y_test = self.load_data(
            PARAMS, PARAMS['dataset']['dataset_ab'], PARAMS['dataset']['dataset_as'],
            self.__config.training_params["validation"], self.__config.training_params["test"],
            self.__config.dataset_settings["nsamples"],
            PARAMS['dataset']['meta-suffix'], PARAMS['dataset']['meta-suffix2']
        )

        # Set model parameters
        model_dir = os.path.join(MODELS_DIR, self.__config.model_id)
        utils.ensure_dir(MODELS_DIR)
        utils.ensure_dir(model_dir)
        model_file = os.path.join(model_dir, self.__config.model_id + MODEL_EXT)
        trained_model = self.__config.get_dict()

        if not os.path.exists(model_file):
            # Construct and save model
            print('Building Network...')
            model = self.build_model(self.__config)
            utils.save_model(model, model_file)
            print(model.summary())

            # Training and validation
            print('\nTraining...')
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            model.fit(
                X_train, Y_train, batch_size=self.__config.training_params['n_minibatch'],
                nb_epoch=self.__config.training_params['n_epochs'], verbose=1,
                validation_data=(X_val, Y_val), callbacks=[early_stopping]
            )

            # Save trained model
            model.save_weights(os.path.join(model_dir, self.__config.model_id + WEIGHTS_EXT))
            utils.save_trained_model(DEFAULT_TRAINED_MODELS_FILE, trained_model)
            print('\nSaving trained model %s in %s...' % (trained_model['model_id'], DEFAULT_TRAINED_MODELS_FILE))
        else:
            model = utils.load_model(model_file)
            model.load_weights(os.path.join(model_dir, self.__config.model_id + WEIGHTS_EXT))
            trained_model = self.__config.get_dict()

        # Predict and evaluate the model for split test data
        print('\nPredicting for split test data...')
        preds = model.predict(X_test)

        r2s = []
        for i,pred in enumerate(preds):
            r2 = r2_score(Y_test[i], pred)
            r2s.append(r2)
        r2 = np.asarray(r2s).mean()
        print('R2 avg: ', r2)

        # Delete used variables
        del X_train, Y_train, X_val, Y_val, X_test, Y_test
        gc.collect()

        # Load trained model and model config
        trained_models = pd.read_csv(DEFAULT_TRAINED_MODELS_FILE, sep='\t')
        model_config = trained_models[trained_models['model_id'] == trained_model['model_id']].to_dict(orient="list")

        # Predict for whole test data
        print('\nPredicting for whole test data...')
        predicted_matrix_map, predictions_index = self.predict(
            model_config, trained_model['model_id'],
            trim_coeff=self.__config.predicting_params['trim_coeff'],
            model=model, fact=PARAMS['dataset']['fact'], dim=PARAMS['dataset']['dim'],
            num_users=PARAMS['dataset']['num_users'], dataset_as=PARAMS['dataset']['dataset_as'],
            meta_source_ab=PARAMS['dataset']['meta-suffix'],
            meta_source_as=PARAMS['dataset']['meta-suffix2']
        )
        print('Prediction is completed.\n')

        # Evaluation
        model_config = trained_models[trained_models["model_id"] == trained_model["model_id"]].to_dict(orient="list")
        model_settings = eval(model_config['dataset_settings'][0])
        model_arch = eval(model_config['model_arch'][0])
        model_training = eval(model_config['training_params'][0])
        str_config = json.dumps(model_settings) + "\n" + json.dumps(model_arch) + "\n" + json.dumps(model_training) + "\n"
        model_settings["loss"] = model_training['loss_func']

        self.evaluate(trained_model['model_id'], model_settings, str_config, predicted_matrix_map, predictions_index)

    def predict(self, model_config, model_id, trim_coeff=0.15, model=False, fact='', dim=200,
                num_users=100, dataset_as='', meta_source_ab='', meta_source_as='', rnd_selection=False):
        '''Predict the model across the whole dataset.'''

        # Initialize variables
        predictions = dict()
        predictions = []
        predictions_index = []
        
        # Load X_meta data
        all_X_meta_ab = self.load_sparse_csr(
            TESTDATA_DIR + '/X_test_%s_%s.npz'
            % (meta_source_ab, PARAMS['dataset']['dataset_ab'])
        )
        all_X_meta_as = np.load(TESTDATA_DIR + '/X_test_%s_%s.npy' % (meta_source_as, dataset_as))
        print("Test data points for: %d" % all_X_meta_ab.shape[0])

        # Load index meta data
        index_meta = open(TESTDATA_DIR + '/items_index_test_%s.tsv' % dataset_as).read().splitlines()
        index_meta_inv = dict()
        for i, item in enumerate(index_meta):
            index_meta_inv[item] = i

        # Predict
        block_step = 1000
        N_test = all_X_meta_ab.shape[0]
        for i in range(0, N_test, block_step):
            x_block = [all_X_meta_ab[i:min(N_test, i + block_step)], all_X_meta_as[i:min(N_test, i + block_step)]]
            preds = model.predict(x_block)
            predictions.extend([prediction for prediction in preds])

        predictions_index = index_meta
        predictions = np.asarray(predictions)
        
        # Save prediction results
        if not os.path.isdir(PREDICTIONS_DIR):
            os.makedirs(PREDICTIONS_DIR)
        np.save(PREDICTIONS_DIR + 'pred_%s' % model_id, predictions)
        fw = open(PREDICTIONS_DIR + 'index_pred_%s.tsv' % model_id, 'w')
        fw.write('\n'.join(predictions_index))
        fw.close()

        # Load user factors based on play counts of each song
        user_factors = np.load(TESTDATA_DIR + '/user_factors_%s_%s_%s.npy' % (fact, dim, dataset_as))
        user_factors = user_factors[:min(num_users, user_factors.shape[0])]        

        # Predict perferences of songs by users
        predicted_matrix_map = normalize(np.nan_to_num(predictions)).dot(user_factors.T).T
        best_song_indices = np.where(predicted_matrix_map == np.amax(predicted_matrix_map, axis=0))[1]
        f = open(PREDICTIONS_DIR + 'track_recommendations.tsv', 'w')
        for i in best_song_indices:
            f.write(predictions_index[i] + '\n')
        f.close()

        return predicted_matrix_map, predictions_index

    def evaluate(self, model_id, model_settings, str_config,
                 predicted_matrix_map, predictions_index):
        '''Evaluate prediction results.'''

        def calc_mapk(actual, predicted, k=10):
            ''' Caculate MAP.'''

            def calc_apk(actual, predicted, k=10):
                if len(predicted) > k:
                    predicted = predicted[:k]

                score = 0.0
                num_hits = 0.0
                for i, p in enumerate(predicted):
                    if p in actual and p not in predicted[:i]:
                        num_hits += 1.0
                        score += num_hits / (i+1.0)

                if not actual:
                    return 0.0
                return score / min(len(actual), k)

            return np.mean([calc_apk(a, p, k) for a, p in zip(actual, predicted)])

        print('Evaluating...')

        # Load ground truth
        index_matrix = open(TESTDATA_DIR + '/items_index_test_%s.tsv' % (model_settings['dataset_as'])).read().splitlines()
        index_matrix_inv = dict((item,i) for i, item in enumerate(index_matrix))
        index_good = [index_matrix_inv[item] for item in predictions_index]

        actual_matrix = self.load_sparse_csr(TESTDATA_DIR + '/matrix_test_%s.npz' % model_settings['dataset_as'])
        actual_matrix_map = actual_matrix[:, :min(model_settings['num_users'], actual_matrix.shape[1])]
        actual_matrix_map = actual_matrix_map[index_good].T.toarray()

        # Write model settings
        if not os.path.exists(RESULTS_DIR): os.makedirs(RESULTS_DIR)
        fw = open(RESULTS_DIR + '/eval_results.txt', 'a')
        fw.write(
            model_id + ', ' + model_settings['dataset_ab'] + ', ' + model_settings['dataset_as'] + ', ' +
            model_settings['configuration'] + ', ' + model_settings['meta-suffix'] + ', ' + 
            model_settings['meta-suffix2'] + '\n' if 'meta-suffix2' in model_settings else '\n'
        )

        # MAP@k
        k = 500
        actual = [list(np.where(actual_matrix_map[i] > 0)[0]) for i in range(actual_matrix_map.shape[0])]
        predicted = list([list(l)[::-1][:k] for l in predicted_matrix_map.argsort(axis=1)])
        map500 = calc_mapk(actual, predicted, k)
        fw.write('MAP@500: %.5f\n' % map500)
        print('MAP@500: %.5f' % map500)

        fw.write('\n')
        fw.write(str_config)
        fw.write('\n')
        fw.close()
