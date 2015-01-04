torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)
require 'cutorch'
require 'cunn'
require 'setup_dummy' -- seed=11, 98.29%, 30.574s
require 'lib/data_prep'
require 'lib/minibatch_sgd'

local function train()
  local train_x,train_y = unpack(torch.load(train_fname))
  local test_x,test_y = unpack(torch.load(test_fname))
  
  if #jitter_params > 0 then 
    train_x, train_y = augment(train_x, train_y, jitter_params)
    collectgarbage()
  end
  
  local model, criterion = create_model()
  model = model:cuda()
  criterion = criterion:cuda()
  local parameters = model:getParameters()
  print('## number of model parameters: ' .. parameters:size(1))
  
  local pre_params = preprocess(train_x)
  torch.save('models/preprocessing_params.t7', params)
  preprocess(test_x, pre_params)
  
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
    torch.save(string.format('models/%s_%d.model', model_id, opt.seed), model)
    if #jitter_params > 0 then
      print(test_augmented(model, params, test_x, test_y, classes))
    else
      print(test(model, params, test_x, test_y, classes))
    end
    collectgarbage()
  end
  
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-seed', 11, 'fixed input seed')
cmd:option('-progress', false, 'show progress bars')
opt = cmd:parse(arg)
print(opt)
torch.manualSeed(opt.seed)
train()