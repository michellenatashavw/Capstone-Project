import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
#import wget

# get csv from api
#url = "https://raw.githubusercontent.com/kristianaryanto/CAPSTONE-PROJECT-GREEN/main/dataset/DKI1.csv"
#filename = wget.download(url)

# Initialize lists

time_step = []
pm10 = []

# Open CSV file
with open('/home/yance/capscrot/CAPSTONE-PROJECT-GREEN/ML/DKI5.csv') as csvfile:
  
  # Initialize reader
  reader = csv.reader(csvfile, delimiter=',')
  
  # Skip the first line
  next(reader)
  
  # Append row and sunspot number to lists
  for row in reader:
    time_step.append(row[1])
    pm10.append(float(row[6]))

# Convert lists to numpy arrays
time = np.array(time_step)
series = np.array(pm10)

# Define the split time
split_time = 300

# Get the train set 
time_train = time[:split_time]
x_train = series[:split_time]

# Get the validation set
time_valid = time[split_time:]
x_valid = series[split_time:]

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
   
    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)
    #dataset.add(tf.data.Dataset.from_tensor_slices(series))
    
    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    
    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels 
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)
    
    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)
    
    return dataset

    # Parameters
window_size = 10
batch_size = 32
shuffle_buffer_size = 1000

# Generate the dataset windows
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)


# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(30, input_shape=[window_size], activation="relu"), 
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Set the learning rate scheduler
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

# Initialize the optimizer
optimizer = tf.keras.optimizers.SGD(momentum=0.9)

# Set the training parameters
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer)

# Train the model
history = model.fit(train_set, epochs=70, callbacks=[lr_schedule])


def model_forecast(model, series, window_size, batch_size):

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    
    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)
    
    # Get predictions on the entire dataset
    forecast = model.predict(dataset)
    
    return forecast

    # Reduce the original series
forecast_series = series[split_time-window_size:-1]

# Use helper function to generate predictions
forecast = model_forecast(model, forecast_series, window_size, batch_size)

# Drop single dimensional axis
results = forecast.squeeze()

# Plot the results
#plot_series(time_valid, ((x_valid, results)))

# Compute the MAE
#print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())
# for call modul
def co(tanggal):
  df = pd.DataFrame(time_valid, columns = ['tanggal'])
  df2 = pd.DataFrame(results, columns = ['Value'])
  df = pd.concat ([df, df2],axis = 1)
  df = df[df['tanggal'] == tanggal]
  return df
  



