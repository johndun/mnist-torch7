require 'torch'
require 'nn'

train_fname = 'data/train.t7'
test_fname = 'data/test.t7'

classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
jitter_params = {}
model_id = 'dummy_model'

sgd_config = {
  learningRate = 1.0,
  finalLearningRate = 0.01, 
  learningRateDecay = 1.0e-6,
  momentum = 0.9,
  xBatchSize = 128, 
  max_epochs = 16
}

function create_model()
  local model = nn.Sequential()
  model:add(nn.Reshape(576))
  model:add(nn.Linear(576, 2048))
  model:add(nn.Tanh())
  model:add(nn.Linear(2048, #classes))
  model:add(nn.LogSoftMax())
  
  local criterion = nn.DistKLDivCriterion()
  return model, criterion
end