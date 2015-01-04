require 'torch'
require 'image'

function load_data(fname)
    local f = torch.load(fname,'ascii')
    local dat = f.data:type(torch.getdefaulttensortype())
    local labs = f.labels
    local labels = torch.Tensor(labs:size(1), 10):zero()
    for i = 1, labs:size(1) do
      labels[i][labs[i]] = 1.0
    end
    return dat, labels
end

function preprocess(x, params)
  local params = params or {}
  
  if #params == 0 then
    params['g_mn'] = x:mean()
    params['g_sd'] = x:std()
  end
  x:add(-params['g_mn'])
  x:mul(1/params['g_sd'])
  
  return params
end

function augment(...)
  local x, y, params, output_size
  local args = {...}
  x = args[1]
  if x:dim() == 3 then
    x = x:resize(1, x:size(1), x:size(2), x:size(3))
  end
  if select('#', ...) == 4 then
    y = args[2]
    params = args[3]
    output_size = args[4]
  elseif select('#', ...) == 3 then
    params = args[2]
    output_size = args[3]
  elseif select('#', ...) == 2 then
    params = args[2]
    output_size = {w = x:size(4), h = x:size(3)}
  end
  
  local ncolors = x:size(2)
  local new_x = torch.Tensor(x:size(1) * #params, 
                             ncolors, output_size['h'], output_size['w'])
  local t = 1
  for i = 1, x:size(1) do
    for j = 1, #params do
      local src = x[i]
      local jit = params[j]
      jit['rotate'] = jit['rotate'] or 0
      jit['hflip'] = jit['hflip'] or 0
      if jit['hflip'] ~= 0 then
        src = image.hflip(src)
      end
      if jit['rotate'] ~= 0 then
        src = image.rotate(src, jit['rotate'])
      end
      src = image.crop(src, jit['x_off'], jit['y_off'], 
                       jit['x_off'] + jit['x_sz'], 
                       jit['y_off'] + jit['y_sz'])
      if jit['x_sz'] ~= output_size['w'] or 
         jit['y_sz'] ~= output_size['h'] then
        src = image.scale(src, output_size['w'], output_size['h'], 'bilinear')
      end
      new_x[t]:copy(src)
      t = t + 1
    end
    collectgarbage()
  end
  
  if y then
    if y:dim() == 1 then
      y = y:resize(1, y:size(1))
    end
    local new_y = torch.Tensor(y:size(1) * #params, y:size(2))
    local t = 1
    for i = 1, y:size(1) do
      for j = 1, #params do
        new_y[t]:copy(y[i])
        t = t + 1
      end
    end
    return new_x, new_y
  end
  
  return new_x
end

-- local jitter = {}
-- local max_rot = math.pi / 18
-- for hflip = 0, 1 do
  -- for rot = -1, 1 do
    -- for x_off = 0, 4, 2 do
      -- for y_off = 0, 4, 2 do
        -- table.insert(jitter, {hflip  = hflip, 
                              -- rotate = max_rot * rot, 
                              -- x_off  = 2 * x_off, 
                              -- y_off  = 2 * y_off, 
                              -- x_sz   = 24, 
                              -- y_sz   = 24 })
        -- table.insert(jitter, {hflip  = hflip, 
                              -- rotate = max_rot * rot, 
                              -- x_off  = x_off, 
                              -- y_off  = y_off, 
                              -- x_sz   = 28, 
                              -- y_sz   = 28 })
      -- end
    -- end
  -- end
-- end
-- local img_size = {w=24, h=24}
-- local fname = '../data/train_32x32.t7'
-- local x, y = load_data(fname)
-- x = x:narrow(1,1,2)
-- y = y:narrow(1,1,2)
-- local x_new, y_new = augment(x, y, jitter, img_size)
-- print(x_new:size())
-- print(y_new:size())
-- local image_tiled = image.toDisplayTensor{input=x_new, padding=4, nrow=18}
-- image.savePNG('../jitter-test.png', image_tiled)