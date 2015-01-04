torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)
require 'cutorch'
require 'cunn'
-- require 'setup_dummy' -- 98.31%, 28.603s
require 'setup_dummy2'
require 'lib/data_prep'
require 'lib/minibatch_sgd'

local function validate()
  local train_size = 50000
  local test_size = 10000
  
  local x, y = unpack(torch.load(train_fname))
  local train_x = x:narrow(1, 1, train_size)
  local train_y = y:narrow(1, 1, train_size)
  local test_x = x:narrow(1, train_size + 1, test_size)
  local test_y = y:narrow(1, train_size + 1, test_size)
  
  if #jitter_params > 0 then
    train_x, train_y = augment(train_x, train_y, jitter_params)
    collectgarbage()
  end
  
  local pre_params = preprocess(train_x)
  preprocess(test_x, pre_params)
  
  local model, criterion = create_model()
  model = model:cuda()
  criterion = criterion:cuda()
  local parameters = model:getParameters()
  print('## number of model parameters: ' .. parameters:size(1))
  
  for epoch = 1, sgd_config.max_epochs do
    if epoch == sgd_config.max_epochs and 
       sgd_config.finalLearningRate ~= 0 then
      sgd_config.learningRateDecay = 0
      sgd_config.learningRate = sgd_config.finalLearningRate
    end
    model:training()
    print('\n# Epoch ' .. epoch)
    print('## train')
    print(minibatch_sgd(model, criterion, train_x, train_y,
          classes, sgd_config))
    print(' + final learning rate: ' .. 
          (sgd_config.learningRate / (1 + sgd_config.learningRateDecay * 
                                          sgd_config.evalCounter)))
    print('## test')
    model:evaluate()
    torch.save(string.format('models/%s_val.model', model_id), model)
    if #jitter_params > 0 then
      print(test_augmented(model, params, test_x, test_y, classes))
    else
      print(test(model, params, test_x, test_y, classes))
    end
    collectgarbage()
  end
  
end

torch.manualSeed(11)
validate()