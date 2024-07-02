# Data
values_per_day = 96
training_days = 7

# Hyperparameters
lstm_units = 30  # Number of LSTM units
sequence_length = values_per_day #* training_days # Sequence length (representing one day)
batch_size = values_per_day #* training_days  # Batch size
num_epochs = 200  # Number of epochs
learning_rate = 0.1  # Learning rate
loss_function = 'mean_squared_error'  # Loss function



# features
features = ['solar_rad_relative', 'ghi_relative', 'precip_1h_relative', 'minute_of_day_relative', 'snow_depth_2h_relative']
# features = ['solar_rad_relative']
target = ['avg_relative']

# other
base_dir = 'data'

