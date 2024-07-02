from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import LambdaCallback
from keras.layers import Input
import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor

import os
import pickle
import logging
import sys

import src.shared.constants as constants
import src.shared.utils as utils
from src.shared.model.site import Site
from src.shared.model.model_data import ModelData
from src.shared.model.model_meta import ModelMeta
from src.shared.aggregation_level import AggregationLevel

class FederatedClient:
        
    def __init__(self, site: Site, value_key: str, data_path: str, server_url: str):
        self.site = site
        self.value_key = value_key
        self.base_path = data_path
        self.server_url = server_url

        self.logging_path = f'{self.base_path}/logs'
        self.models_path = f'{self.base_path}/models'
        self.predictions_path = f'{self.base_path}/predictions'

        os.makedirs(self.models_path, exist_ok=True)

        self._setup_logging()

        self.model_data = self._get_local_model()

        if self.model_data is None:
            model_data = ModelData(ModelMeta(), utils.get_model().get_weights())
            self._save_local_model(AggregationLevel.site, model_data)
            self.model_data = model_data
            
        self.is_registrered = False
        self._register_client()

    def _preprocess_data(self, data):
        if data is None or data.empty:
            logging.error(f"No data found for value_key: {self.value_key}. Exiting function.")
            return None

        if not any(data['value_key'] == self.value_key):
            logging.error(f"No values found with value_key: {self.value_key}. Exiting function.")
            return None

        default_columns = ['time', 'avg', 'solar_rad', 'precip', 'ghi', 'temp', 'snow_depth']
        time_shifted_columns = ['solar_rad_1h', 'precip_1h', 'ghi_1h', 'temp_1h', 'snow_depth_1h', 'solar_rad_2h', 'precip_2h', 'ghi_2h', 'temp_2h', 'snow_depth_2h', 'avg_24h']
        columns = default_columns + time_shifted_columns
        
        data = data[columns].copy()  # Ensure we're working with a copy

        # Adapt data based on conditions
        data.loc[data['snow_depth_1h'] >= 1, ['solar_rad', 'ghi']] = 0

        # Compute and scale 'avg_relative', 'solar_rad', and 'ghi'
        data['avg_relative'] = data['avg'] / (float(self.site.kwp) * 1000)
        data['avg_24h_relative'] = data['avg_24h'] / (float(self.site.kwp) * 1000)

        # divide solar_rad by max (~1000 W/m^2) to get relative value
        data['solar_rad_relative'] = data['solar_rad'] / 1000
        data['solar_rad_1h_relative'] = data['solar_rad_1h'] / 1000
        data['solar_rad_2h_relative'] = data['solar_rad_2h'] / 1000

        # divide ghi by 1000 to get relative value
        data['ghi_relative'] = data['ghi'] / 1000
        data['ghi_1h_relative'] = data['ghi_1h'] / 1000
        data['ghi_2h_relative'] = data['ghi_2h'] / 1000

        # divide snow depth by 1000 to get relative value
        data['snow_depth_relative'] = data['snow_depth'] / 1000
        data['snow_depth_1h_relative'] = data['snow_depth_1h'] / 1000
        data['snow_depth_2h_relative'] = data['snow_depth_2h'] / 1000

        # divide precip by 15 to get relative value
        data['precip_relative'] = data['precip'] / 15
        data['precip_1h_relative'] = data['precip_1h'] / 15
        data['precip_2h_relative'] = data['precip_2h'] / 15

        # divide temp by 40 to get relative value
        data['temp_relative'] = data['temp'] / 40
        data['temp_1h_relative'] = data['temp_1h'] / 40
        data['temp_2h_relative'] = data['temp_2h'] / 40

        # add minute of the day
        data['minute_of_day'] = data['time'].dt.hour * 60 + data['time'].dt.minute
        data['minute_of_day_relative'] = data['minute_of_day'] / 1440

        # add day of year
        data['day_of_year'] = data['time'].dt.dayofyear
        data['day_of_year_relative'] = data['day_of_year'] / 365

        data = data.sort_values('time')

        # drop rows which are defective
        for row in data.iterrows():
            # check if avg is negative or higher than kwp * 1000
            if row[1]['avg'] < 0 or row[1]['avg'] > (float(self.site.kwp) * 1000) or row[1]['avg_24h'] < 0 or row[1]['avg_24h'] > (float(self.site.kwp) * 1000):
                # delete row
                data = data.drop(row[0])

            # check if one of the constants.features is NaN
            if any(row[1][constants.features].isnull()):
                # delete row
                data = data.drop(row[0])
        
        return data


    def _postprocess_data(self, data, predictions):
        if predictions is None or len(predictions) != constants.values_per_day * constants.training_days:
            logging.error(f"Data is incomplete. Exiting training function.")
            return None

        # predictions = self.scaler.inverse_transform(predictions_1d)  # Inverse transform predictions to original scale
        predictions = predictions.flatten()

        # multiply by kwp to get actual values
        predictions = predictions * (float(self.site.kwp) * 1000)

        # set values between between 5 AM and 10 PM to 0
        for i in range(len(data)):
            time = data['time'].iloc[i]
            if time.hour < 5 or time.hour > 22:
                predictions[i] = 0
        
         # set negative values to 0
        predictions[predictions < 0] = 0

        # set to kwp * 1000 if higher than kwp * 1000
        predictions[predictions > (float(self.site.kwp) * 1000)] = float(self.site.kwp) * 1000
        return predictions

    def _train_model(self, level: AggregationLevel, model_data: ModelData, dates, data) -> ModelData:
        features = constants.features
        target = constants.target
        data_x = np.array(data[features])
        data_y = np.array(data[target])

        # Reshape for LSTM input
        data_x = data_x.reshape((data_x.shape[0], -1, len(features)))
        data_y = data_y.reshape((data_y.shape[0], len(target)))  # Target variable does not need additional dimensions

        model = utils.get_model()
        model.set_weights(model_data._get_weigths_np())

        model.fit(
            data_x[:-1], data_y[1:], 
            epochs=constants.num_epochs, 
            batch_size=constants.batch_size, 
            verbose=0,  # Suppress detailed output
            callbacks=[
                LambdaCallback(
                    on_epoch_end=lambda epoch, logs: logging.info(
                        f'Date: {dates[0]} - {dates[-1]}, Aggregation Level: {level} Epoch: {epoch+1}, Loss: {format(logs["loss"], ".10f")}'
                    )
                )
            ]
        )

        model_meta = ModelMeta(
            num_samples_learned=model_data.num_samples_learned + len(data),
            num_epochs_learned=model_data.num_epochs_learned + constants.num_epochs,
            num_round=model_data.num_round + 1,
            num_samples_epochs_learned=model_data.num_samples_epochs_learned + len(data) * constants.num_epochs,
            learned_dates=model_data.learned_dates + [date.strftime('%Y-%m-%d') for date in dates]
        )

        model_delat_meta = ModelMeta(
            num_samples_learned=len(data),
            num_epochs_learned=constants.num_epochs,
            num_round=1,
            num_samples_epochs_learned=len(data) * constants.num_epochs,
            learned_dates=[date.strftime('%Y-%m-%d') for date in dates]
        )

        return ModelData(model_meta, model.get_weights()), model_delat_meta

    def _predict(self, model_data: ModelData, data):
        if data is None or len(data) != constants.values_per_day * constants.training_days:
            logging.info(f"Data is incomplete. Exiting training function.")
            return None
        
        features = constants.features
        data_x = np.array(data[features])
        data_x = data_x.reshape((data_x.shape[0], -1, len(features)))

        model = utils.get_model()
        model.set_weights(model_data._get_weigths_np())

        # Predictions
        predictions = model.predict(data_x)

        return predictions


    def _compare_save_predictions(self, data, predictions_local_model, predictions_cluster_model, predictions_global_model):
        if predictions_local_model is None or len(predictions_local_model) != len(data) or predictions_cluster_model is None or len(predictions_cluster_model) != len(data) or predictions_global_model is None or len(predictions_global_model) != len(data):
            logging.error(f"Predictions or data is incomplete. Exiting function.")
            return None
        
        avg_local_global = np.zeros_like(predictions_local_model)

        for i in range(len(data)):
            avg_local_global[i] = np.mean([predictions_local_model[i], predictions_global_model[i]])
        
        # Create DataFrame with predicted values
        df = pd.DataFrame({
            'site_id': self.site.site_id,
            'value_key': self.value_key,
            'time': data['time'].values,
            'avg_predicted_local': predictions_local_model,
            'avg_predicted_cluster': predictions_cluster_model,
            'avg_predicted_global': predictions_global_model,
            'avg_predicted_avg_local_global': avg_local_global,
            'avg': data['avg'].values,
            'ghi': data['ghi'].values,
            'solar_rad': data['solar_rad'].values,
            'temp': data['temp'].values,
            'precip': data['precip'].values,
            'snow_depth': data['snow_depth'].values,
            'kwp': self.site.kwp
        })

        return df

    def _register_client(self) -> bool:
        if self.is_registrered is True:
            return True

        if self.server_url is None:
            logging.error("No server URL provided. Exiting function.")
            return False
        
        try:
            logging.info(f"Registering client for site {self.site.site_id} on server {self.server_url}.")
            response = requests.post(f'{self.server_url}/site/{self.site.site_id}/register')
            if response.status_code != 200:
                raise 
        except:
            logging.error("Failed to register client. Exiting function.")
            self.is_registrered = False
            return False

        self.is_registrered = True
        return True
    
    def _save_local_model(self, lvl: AggregationLevel, model_data: ModelData):
        model_data_path = f'{self.models_path}/{self.value_key}_{lvl}.pkl'

        with open(model_data_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def _get_local_model(self, lvl: AggregationLevel = 'site') -> ModelData:
        model_data_path = f'{self.models_path}/{self.value_key}_{lvl}.pkl'
        model_data = None

        if os.path.exists(model_data_path):
             with open(model_data_path, 'rb') as f:
                model_data = pickle.load(f)
             
        return model_data
    
    def _get_server_model(self, lvl: AggregationLevel = 'site') -> ModelData:
        if self._register_client() is False:
            logging.error("Failed to register client. Exiting function.")
            return None
        
        try:
            response = requests.get(f'{self.server_url}/site/{self.site.site_id}/model/{lvl}')
            if response.status_code != 200:
                raise
            
            model_data = ModelData.from_json(response.json().get('model_data', None))
        except:
            logging.error(f"Failed to get {lvl}-model. Exiting function.")
            return None

        return model_data

    def _propagate_model(self, lvl: AggregationLevel = 'site', model_data: ModelData = None, model_delta_meta: ModelMeta = None):
        if self._register_client() is False:
            logging.error("Failed to register client. Exiting function.")
            return None
            
        try:
            response = requests.post(f'{self.server_url}/site/{self.site.site_id}/model/{lvl}', json={'model_data': model_data.to_json(), 'model_delta_meta': model_delta_meta.to_json()})
            if response.status_code != 200:
                raise Exception(f"Failed to propagate {lvl}-model for site {self.site.site_id}. Exiting function.")
        except:
            logging.error(f"Failed to propagate {lvl}-model for site {self.site.site_id}. Exiting function.")

    
    def _get_model_delta(self, model_data_old: ModelData, model_data_new: ModelData) -> ModelData:
        model_delta_meta = ModelMeta(
            num_samples_learned=model_data_new.num_samples_learned - model_data_old.num_samples_learned,
            num_epochs_learned=model_data_new.num_epochs_learned - model_data_old.num_epochs_learned,
            num_round=model_data_new.num_round - model_data_old.num_round,
            num_samples_epochs_learned=model_data_new.num_samples_epochs_learned - model_data_old.num_samples_epochs_learned,
            learned_dates=model_data_new.learned_dates
        )

        old_model_weigths = model_data_old._get_weigths_np()
        new_model_weigths = model_data_new._get_weigths_np()

        # calculate delta weights
        for i in range(len(old_model_weigths)):
                new_model_weigths[i] -= old_model_weigths[i]

        model_delta = ModelData(model_delta_meta, new_model_weigths)

        return model_delta

    def check_dates(self, dates):
        if not utils.are_subsequent(dates):
            logging.error(f"Data for dates {utils.dates_to_daystrings(dates)} is not subsequent. Exiting function.")
            return False

        if len(dates) != constants.training_days:
            logging.error(f"Data for dates {utils.dates_to_daystrings(dates)} is incomplete. Exiting function.")
            return False
        
        return True

    def _train_and_save_model(self, model_data, dates, data, level):
        if model_data is None:
            logging.error(f"No model data found for {level}. Exiting function.")
            return None, level

        updated_model_data, model_delta_meta = self._train_model(level, model_data, dates, data)
        return updated_model_data, model_delta_meta, level

    def process_data(self, data):
        if data is None or data.empty:
            logging.error(f"No data found for value_key: {self.value_key}. Exiting function.")
            return None

        dates = data['time'].dt.date.unique()
        dates.sort()

        if not self.check_dates(dates):
            return None
        
        if data is None or len(data) != constants.values_per_day * constants.training_days:
            logging.info(f"Data for date {utils.dates_to_daystrings(dates)} is incomplete. Exiting training function.")
            return None
        
        # check if some dates are in learned_dates
        for date in dates:
            if date.strftime('%Y-%m-%d') in self.model_data.learned_dates:
                logging.info(f'Already learned data for date {date}, ignore ...')
                return None


        data = self._preprocess_data(data)

        if data is None:
            logging.error(f"No data found for date {utils.dates_to_daystrings(dates)}. Exiting function.")
            return None

        logging.info(f'Data complete and well shaped, start training ...')

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for level in [AggregationLevel.site, AggregationLevel.cluster, AggregationLevel.global_]:
                model_data = self._get_local_model(level) if level == AggregationLevel.site else self._get_server_model(level)
                logging.info(f"Start training for {level} ...")
                future = executor.submit(self._train_and_save_model, model_data, dates, data, level)
                futures.append(future)

            for future in futures:
                updated_model_data, model_delta_meta, level = future.result()

                if updated_model_data is not None:
                    if level != AggregationLevel.site:
                        self._propagate_model(level, updated_model_data, model_delta_meta)
                    else:
                        self._save_local_model(level, updated_model_data)
    
    def predict_data(self, data):
        if data is None or data.empty:
            logging.error(f"No data found for value_key: {self.value_key}. Exiting function.")
            return None

        dates = data['time'].dt.date.unique()
        dates.sort()

        if not self.check_dates(dates):
            return None
        
        data = self._preprocess_data(data)

        if data is None:
            logging.error(f"No data found for date {utils.dates_to_daystrings(dates)}. Exiting function.")
            return None

        logging.info(f'Data complete and well shaped, start predictions ...')


        
        site_model = self._get_local_model(AggregationLevel.site)
        cluster_model = self._get_server_model(AggregationLevel.cluster)
        global_model = self._get_server_model(AggregationLevel.global_)

        predictions = self._predict(site_model, data)
        predictions_cluster_model  = self._predict(cluster_model, data)
        predictions_global_model = self._predict(global_model, data)
        predictions = self._postprocess_data(data, predictions)
        predictions_cluster_model = self._postprocess_data(data, predictions_cluster_model)
        predictions_global_model = self._postprocess_data(data, predictions_global_model)
        
        return self._compare_save_predictions(data, predictions, predictions_cluster_model, predictions_global_model)


    def _setup_logging(self):
        os.makedirs(self.logging_path, exist_ok=True)
        log_file = f'{self.logging_path}/{self.value_key}.log'

        # Clear any existing log handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Set up logging to file and console
        logging.basicConfig(level=logging.INFO,
                            format=f'%(asctime)s - {self.site.site_id} - {self.value_key} - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_file),
                                logging.StreamHandler(sys.stdout)
                            ])