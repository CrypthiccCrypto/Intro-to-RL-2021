# Load Dependencies #
import torch
import torch.nn as nn
import tensorflow as tf
import cv2
#####################
from gym.core import ObservationWrapper
from gym.spaces import Box

def _to_gray_scale(rgb, channel_weights=[0.15, 0.8, 0.05]):
    return np.float32(np.array([np.dot(rgb[..., :3], channel_weights)]))

def preprocessing(batch):
    pbatch = [] #stores preprocessed batch

    for i in range(0, batch.shape[0]):
      tmp = batch[i][0:84, 6:90]
      tmp = cv2.resize(tmp, (64, 64))
      tmp = _to_gray_scale(tmp)/255
      pbatch.append(tmp)

    return np.array(pbatch) 

def output_preprocessing(output):
  result = []
  for j in range(0, output.shape[0]):
    tmp = []  
    for i in range(0, 8):
      tmp.append(1 if np.array_equal(y_train[j], DISC_ACTION_SPACE[i]) else 0)
    result.append(tmp)
  return torch.from_numpy(np.array(result))

def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride  + 1

class ILAgent(nn.Module):
    def __init__(self, state_shape, n_actions):

        super().__init__()
        self.n_actions = n_actions
        self.state_shape = state_shape
        length = state_shape[1]

        self.ReLU = nn.ReLU()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 2)
        length = conv2d_size_out(length, 3, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 2)
        length = conv2d_size_out(length, 3, 2)
        self.lin1 = nn.Linear(32*length*length, 256)
        self.lin2 = nn.Linear(256, n_actions)

    def get_action(self, state_t):
        state_t = torch.from_numpy(state_t)
        action = state_t
        action = self.conv1(action)
        action = self.ReLU(action)
        action = self.conv2(action)
        action = self.ReLU(action)
        action = self.flatten(action)
        action = self.lin1(action)
        action = self.ReLU(action)
        action = self.lin2(action)

        return action