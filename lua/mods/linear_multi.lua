-- linear layer with multple parameters
-- which param to use is set by model_ids
function linear_multi(sz_in, sz_out, model_ids, input, nmodels)
	if nmodels == 1 then
		return nn.Linear(sz_in, sz_out)(input)
	end
    local weight_lut = nn.LookupTable(nmodels, sz_in * sz_out)(model_ids)
    weight_lut.data.module.updateGradInput = function(self, input) return input end
    local bias_lut = nn.LookupTable(nmodels, sz_out)(model_ids)
    bias_lut.data.module.updateGradInput = function(self, input) return input end
    local weight_view = nn.View(sz_out, sz_in):setNumInputDims(1)(weight_lut)
    input = nn.View(-1, 1):setNumInputDims(1)(input)
    local out = nn.MM(false, false)({weight_view, input})
    out = nn.View(-1):setNumInputDims(2)(out)
    out = nn.CAddTable()({out, bias_lut})
    out.weight_lut = weight_lut
    out.bias_lut = bias_lut
    return out
end

function linear_multi_test()
	local l1 = nn.Linear(10, 20)
	local l2 = nn.Linear(10, 20)
	local x = torch.rand(3, 10)
	local y1 = l1:forward(x)
	local y2 = l2:forward(x)
	local input = nn.Identity()()
	local ids = nn.Identity()()
	local lm = linear_multi(10, 20, ids, input, 2)
	local w = lm.weight_lut.data.module
	local b = lm.bias_lut.data.module
	w.weight[1]:copy(l1.weight:view(-1))
	w.weight[2]:copy(l2.weight:view(-1))
	b.weight[1]:copy(l1.bias)
	b.weight[2]:copy(l2.bias)

	local model = nn.gModule({input, ids}, {lm})
	local y = model:forward({x, torch.Tensor({1, 2, 1})})
	assert(y:size(1) == 3)
	assert(y:size(2) == 20)
	assert(y:dim() == 2)
	assert(y1[1]:add(-1, y[1]):abs():sum() < 1e-3)
	assert(y2[2]:add(-1, y[2]):abs():sum() < 1e-3)
	assert(y1[3]:add(-1, y[3]):abs():sum() < 1e-3)
end
