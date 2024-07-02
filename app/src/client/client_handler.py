import os
import pandas as pd
import fcntl

from src.client.federated_client import FederatedClient
import src.shared.constants as constants
from src.shared.model.site import Site
from src.shared.model.model_data import ModelData

if __name__ == "__main__":
    data_path = os.getenv('DATA_PATH')
    test_data_path = os.getenv('TEST_DATA_PATH')
    shared_data_path = os.getenv('SHARED_DATA_PATH')
    site_id = os.getenv('SITE_ID')
    value_key = os.getenv('VALUE_KEY')
    server_address = os.getenv('SERVER_ADDRESS')
    server_port = os.getenv('SERVER_PORT')
    run_training = os.getenv('RUN_TRAINING').lower() in ('true', '1', 'True')
    run_prediction = os.getenv('RUN_PREDICTION').lower() in ('true', '1', 'True')

    server_url = f'http://{server_address}:{server_port}'
    
    site_info = pd.read_csv(os.path.join(test_data_path, f'site_info.csv'), header=None).set_index(0).squeeze().to_dict()

    # check if data file exists
    if not os.path.exists(os.path.join(test_data_path, f'{value_key}.csv')):
        print(f"Data file {value_key}.csv does not exist")
        exit(1)

    data = pd.read_csv(os.path.join(test_data_path, f'{value_key}.csv'))
    training_windows = pd.read_csv(os.path.join(test_data_path, f'training_windows.csv'))
    test_windows = pd.read_csv(os.path.join(test_data_path, f'test_windows.csv'))
    
    data['time'] = pd.to_datetime(data['time'])

    training_windows = training_windows.applymap(lambda s: pd.to_datetime(s).date())
    test_windows = test_windows.applymap(lambda s: pd.to_datetime(s).date())

    site = Site(
        site_id=site_info['site_id'],
        cluster=int(site_info['cluster']),
        lat=float(site_info['lat']),
        lng=float(site_info['lng']),
        zip=int(site_info['zip']),
        country=site_info['country'],
        kwp=float(site_info['kwp']),
        weather_data=site_info['weather_data'].lower() == 'True'  # Convert string to boolean
    )


    client = FederatedClient(site, value_key, data_path, server_url)

    if run_training:
        for window in training_windows.values:
            window_data = None

            # Select data for the current window
            mask = data['time'].dt.date.isin(window)
            if mask.any():
                print("Setting window data to training data")
                window_data = data[mask]

            client.process_data(window_data)

    if run_prediction:
        for window in test_windows.values:
            window_data = None

            # Select data for the current window
            mask = data['time'].dt.date.isin(window)
            if mask.any():
                print("Setting window data to test data")
                window_data = data[mask]

            df = client.predict_data(window_data)

            if df is not None:
                with open(f'{shared_data_path}/actual_vs_predicted_all.csv', 'a') as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    df.to_csv(f, header=f.tell()==0, index=False)
                    fcntl.flock(f, fcntl.LOCK_UN)
