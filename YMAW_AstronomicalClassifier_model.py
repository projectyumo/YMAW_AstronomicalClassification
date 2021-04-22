import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

import random

class NeuralNetwork(nn.Module):
  def __init__(self, size_inputs, size_outputs, nodes=24):
    super(NeuralNetwork, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(size_inputs, nodes),
      nn.BatchNorm1d(nodes),
      nn.LeakyReLU(),
      nn.Linear(nodes, int(nodes/2)),
      nn.BatchNorm1d(int(nodes/2)),
      nn.LeakyReLU(),
      nn.Linear(int(nodes/2), size_outputs),
      nn.Sigmoid()
    )

  def forward(self, inputs):
    results = self.net(inputs)
    return results

class AstronomicalClassifier:
  def __init__(self):
    self.train_data = pd.read_csv('reduced_train_data.csv')
    self.train_metadata = pd.read_csv('reduced_train_metadata.csv')

    self.aggs = {'mjd'                   : ['min', 'max', 'size'],
                 'flux'                  : ['min', 'max', 'mean', 'median', 'std', 'skew'],
                 'flux_err'              : ['min', 'max', 'mean', 'median', 'std', 'skew'],
                 'detected'              : ['mean'],
                 'flux_fract_unc'        : ['sum', 'skew'],
                 'flux_by_flux_ratio_sq' : ['sum', 'skew']}

    self.feature_engineer()

  def feature_engineer(self):
    # Feature Engineering/Data Processing
    self.train_data['flux_fract_unc'] = self.train_data['flux']/self.train_data['flux_err']
    self.train_data['flux_by_flux_ratio_sq'] = self.train_data['flux']*np.power(self.train_data['flux_fract_unc'], 2)

    # Aggregate Data
    agg_data = self.train_data.groupby(['object_id']).agg(self.aggs)
    new_columns = [k + '_' + agg for k in self.aggs.keys() for agg in self.aggs[k] ]

    agg_data.columns = new_columns
    agg_data['mjd_diff'] = agg_data['mjd_max'] - agg_data['mjd_min']
    agg_data['flux_diff'] = agg_data['flux_max'] - agg_data['flux_min']
    agg_data['flux_diff_mean'] = agg_data['flux_diff']/agg_data['flux_mean']
    agg_data['flux_by_time'] = agg_data['flux_diff']/agg_data['mjd_diff']
    agg_data = agg_data.reset_index()

    # Merge aggregated data with metadata
    all_data = agg_data.merge(right=self.train_metadata,
                              how='outer',
                              on='object_id')

    # Convert targets to be compatible with model.
    y = all_data['target']
    classes = sorted(y.unique())

    class_map = dict()
    for i, val in enumerate(classes):
      class_map[val] = i

    def to_categorical(y, num_classes):
      """1-hot encode"""
      return np.eye(num_classes, dtype='uint8')[y]

    y_map = np.array([class_map[val] for val in y])
    y_categorical = to_categorical(y_map, 14)

    #Remove Unnecessary Data
    del all_data['object_id'], all_data['distmod'], all_data['ra'], all_data['decl'], all_data['gal_l'], all_data['gal_b'], all_data['target']

    #Remove Invalid/Empty Data
    train_mean = all_data.mean(axis=0)
    all_data.fillna(train_mean, inplace=True)

    self.all_data      = all_data
    self.y_map         = y_map
    self.y_categorical = y_categorical

  def make_NN(self,
              num_epochs=100,
              nodes=50,
              lr=0.005):
    # Initialize Model's Neural Network, Loss Function, and Optimizer
    size_inputs  = self.all_data.shape[1]
    size_outputs = self.y_categorical.shape[1]

    self.myNN = NeuralNetwork(size_inputs, size_outputs, nodes)
    self.criterion = nn.BCELoss()
    self.optimizer = torch.optim.Adam(self.myNN.parameters(), lr=lr)
    self.num_epochs = num_epochs # Training Length
    self.batch_size = 100

  def train_test(self, inp, labels, mode='train'):
    self.optimizer.zero_grad()
    preds = self.myNN(inp)
    error = self.criterion(preds, labels)

    if mode == 'train':
      error.backward()
      self.optimizer.step()

    return preds, error.item()

  def preprocess(self):
    xTrain, xTest, yTrain, yTest = train_test_split(self.all_data, self.y_categorical, test_size=0.2)
    ss = StandardScaler()

    self.xTrain = ss.fit_transform(xTrain)
    self.xTest  = ss.transform(xTest)
    self.yTrain = yTrain
    self.yTest  = yTest

    # Create Tensor Objects
    self.train_x = torch.FloatTensor(self.xTrain)
    self.train_y = torch.FloatTensor(self.yTrain)
    self.test_x  = torch.FloatTensor(self.xTest)
    self.test_y  = torch.FloatTensor(self.yTest)

  def train_NN(self):
    self.preprocess()
    # Store losses
    self.train_losses = []
    self.test_losses = []

    # Train and test at every epoch
    for i in range(self.num_epochs):
      for batch in range(0, self.train_x.shape[0], self.batch_size):
        epoch_losses = []
        _, train_loss = self.train_test(self.train_x[batch:batch+self.batch_size], self.train_y[batch:batch+self.batch_size], mode='train')
        epoch_losses.append(train_loss)
      self.train_losses.append(np.sum(epoch_losses))

      _, test_loss = self.train_test(self.test_x, self.test_y, mode='test')
      self.test_losses.append(test_loss)

      print("Epoch: ", i, ", Training Loss:", train_loss, "Test Loss:", test_loss)

  def visualize_training(self):
    # Simple plot for losses
    plt.figure(figsize=(16,10))
    plt.title('Training Progress')
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.plot(self.train_losses, label='training loss')
    plt.plot(self.test_losses, label='test loss')
    plt.legend()
    plt.show()

  def preview_data(self):
    return self.train_data.head()

  def get_example(self, extype='Supernova Type I'):
    if extype.lower() == 'supernova type i': target=62
    elif extype.lower() == 'supernova type ii': target=42
    elif extype.lower() == 'binary star': target=16
    elif extype.lower() == 'active galactic nuclei': target=88

    valid_oids = self.train_metadata[self.train_metadata.target == target].object_id.values
    sample = random.choice(valid_oids)

    for pb in range(1, 6):
      sns = self.train_data[self.train_data.object_id == sample][self.train_data.passband == pb]
      plt.figure(figsize=(16,10))
      plt.title(f'{extype} Light Curve Example')
      plt.xlabel('Time (Mjd)')
      plt.ylabel('Flux')
      plt.scatter(sns['mjd'], sns['flux'], label=str(pb))
      plt.legend()
      plt.show()
