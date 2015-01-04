require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(4)

require 'lib/data_prep'
local train0_fname = 'data/train_32x32.t7'
local train_fname = 'data/train.t7'
local test0_fname = 'data/test_32x32.t7'
local test_fname = 'data/test.t7'
local train_x,train_y = load_data(train0_fname)
local test_x,test_y = load_data(test0_fname)

local img_size = {w=24, h=24}
local jit = {{x_off=2, y_off=2, x_sz=28, y_sz=28}}
train_x = augment(train_x, jit, img_size)
test_x = augment(test_x, jit, img_size)

torch.save(train_fname, {train_x,train_y})
torch.save(test_fname, {test_x,test_y})