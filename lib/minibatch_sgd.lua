require 'optim'
require 'xlua'

function minibatch_sgd(model, criterion, 
                       train_x, train_y,
                       classes, config)
  local parameters, gradParameters = model:getParameters()
  local confusion = optim.ConfusionMatrix(classes)
  config = config or {}
  local batch_size = config.xBatchSize or 12
  local shuffle = torch.randperm(train_x:size(1))
  local c = 1
  for t = 1, train_x:size(1), batch_size do
    if t + batch_size > train_x:size(1) then
      break
    end
    if opt.progress then
      xlua.progress(t, train_x:size(1))
    end
    local inputs = torch.Tensor(batch_size,
                                train_x:size(2),
                                train_x:size(3),
                                train_x:size(4))
    local targets = torch.Tensor(batch_size,
                                 train_y:size(2))
    for i = 1, batch_size do
      inputs[i]:copy(train_x[shuffle[t + i - 1]])
      targets[i]:copy(train_y[shuffle[t + i - 1]])
    end
    inputs = inputs:cuda()
    targets = targets:cuda()

    local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()
      local output = model:forward(inputs)
      local f = criterion:forward(output, targets)
      local df_do = criterion:backward(output, targets)
      confusion:batchAdd(output, targets)      
      model:backward(inputs, df_do)
      return f, gradParameters
    end
    
    optim.sgd(feval, parameters, config)
    c = c + 1
    if c % 1000 == 0 then
      collectgarbage('collect')
    end
  end
  if opt.progress then
    xlua.progress(train_x:size(1), train_x:size(1))
  end

  return confusion
end

function test_augmented(model, params, test_x, test_y)
  local confusion = optim.ConfusionMatrix(classes)
  for i = 1, test_x:size(1) do
    if opt.progress then
      xlua.progress(i, test_x:size(1))
    end
    local preds = torch.Tensor(10):zero()
    local x = augment(test_x[i], jitter_params)
    local z = model:forward(x:cuda()):float()
    for j = 1, x:size(1) do
      preds = preds + z[j]
    end
    preds:div(x:size(1))
    confusion:add(preds, test_y[i])
  end
  if opt.progress then
    xlua.progress(test_x:size(1), test_x:size(1))
  end
  return confusion
end

function test(model, params, test_x, test_y)
  local confusion = optim.ConfusionMatrix(classes)
  local batch_size = 100 -- test_x:size(1) % batch_size == 0
  for t = 1, test_x:size(1), batch_size do
    if opt.progress then
      xlua.progress(t, test_x:size(1))
    end
    local inputs = torch.Tensor(batch_size,
                                test_x:size(2),
                                test_x:size(3),
                                test_x:size(4))
    local targets = torch.Tensor(batch_size,
                                 test_y:size(2))
    for i = 1, batch_size do
      inputs[i]:copy(test_x[t + i - 1])
      targets[i]:copy(test_y[t + i - 1])
    end
    inputs = inputs:cuda()
    targets = targets:cuda()

    local output = model:forward(inputs)
    for k = 1, output:size(1) do
      confusion:add(output[k], targets[k])
    end
  end
  if opt.progress then
    xlua.progress(test_x:size(1), test_x:size(1))
  end
  return confusion
end