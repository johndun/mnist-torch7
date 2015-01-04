require 'torch'
require 'nn'
require 'cunn'


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
  max_epochs = 20
}

function create_model()
  local model = nn.Sequential() 
  model:add(nn.SpatialConvolutionMM(3, 128, 5, 5, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.SpatialConvolutionMM(128, 256, 5, 5, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
  model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
  model:add(nn.SpatialConvolutionMM(256, 512, 4, 4, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.SpatialConvolutionMM(512, 1024, 2, 2, 1, 1))
  model:add(nn.ReLU())
  model:add(nn.Dropout(0.5))
  model:add(nn.SpatialConvolutionMM(1024, #classes, 1, 1, 1, 1))
  model:add(nn.Reshape(10))
  model:add(nn.LogSoftMax())
  local criterion = nn.DistKLDivCriterion()
  return model, criterion
end