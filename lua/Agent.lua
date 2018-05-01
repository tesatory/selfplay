require('nn')
require('nngraph')
require('optim')
paths.dofile('mods/Switch.lua')
paths.dofile('mods/Sampler.lua')
paths.dofile('mods/GaussianSampler.lua')

local Agent = torch.class('Agent')

function Agent:__init(opts)
    self.opts = opts

    self.model = self:build_model()
    self:build_model_clones(self.opts.max_steps+1)

    self.paramx, self.paramdx = self.model:parameters()
    self:init_params()

    self:build_losses()
    self.optim_state = {}
end

function Agent:init_params()
    if self.opts.init_std > 0 then
        for _, x in pairs(self.paramx) do
            x:normal(0, self.opts.init_std)
        end
    end
    if self.opts.encoder_lut then
        self:zero_encoder_nil()
    end
end

function Agent:zero_encoder_nil()
    -- zero NIL embedding
    self.model.mods['encoder_lut'].weight[self.encoder_lut_nil]:zero()
end

function Agent:build_model_clones(N)
    self.model_clones = {}
    for t = 1, N do
        self.model_clones[t] = self:build_model()
        self.model_clones[t]:share(self.model,
            'weight','bias','gradWeight','gradBias')
    end
end

function Agent:build_losses()
    self.value_loss = nn.MSECriterion()
    self.value_loss.sizeAverage = false
end

function Agent:nonlin()
    if self.opts.nonlin == 'tanh' then
        return nn.Tanh()
    elseif self.opts.nonlin == 'relu' then
        return nn.ReLU()
    elseif self.opts.nonlin == 'none' then
        return nn.Identity()
    else
        error('wrong nonlin')
    end
end

-- network for converting raw input to vector
function Agent:build_encoder(hidsz)
    if self.opts.rllab then
        return nn.Linear(self.opts.rllab_in_dim, hidsz)
    end
    local size = self.opts.visibility*2+1
    local in_dim = size^2 * self.opts.nwords
    in_dim = in_dim + self.opts.max_info * self.opts.nwords
    if self.opts.encoder_lut then
        in_dim = in_dim + 1 -- for NIL padding
        self.encoder_lut_nil = in_dim
        self.opts.encoder_lut_nil = in_dim -- TODO: bit hacky
        local m = nn.LookupTable(in_dim, hidsz)
        local s = nn.Sequential()
        s:add(m)
        s:add(nn.Sum(2))
        s:add(nn.Add(hidsz)) -- bias
        return s, m
    else
        local m = nn.Linear(in_dim, hidsz)
        return m
    end
end

function Agent:build(name, mods)
    if name == 'obs' or name == 'prev_hid' then
        -- inputs to the model
        mods[name] = nn.Identity()()
        mods.inputs[#mods.inputs+1] = name
    elseif name == 'pre_hid' then
        local enc, lut = self:build_encoder(self.opts.hidsz)
        if lut then mods['encoder_lut'] = lut end
        mods.pre_hid = enc(self:get('obs', mods))
        if self.opts.recurrent then
            local hid2hid = nn.Linear(self.opts.hidsz, self.opts.hidsz)(self:get('prev_hid', mods))
            mods.pre_hid = nn.CAddTable()({mods.pre_hid, hid2hid})
        end
    elseif name == 'hidstate' then
        -- internal hidden state
        mods.hidstate = self:nonlin()(self:get('pre_hid', mods))
        for l = 2, self.opts.nlayers do
            mods.linear_layers = mods.linear_layers or {}
            mods.linear_layers[l-1] = nn.Linear(self.opts.hidsz, self.opts.hidsz)
            mods.hidstate = self:nonlin()(mods.linear_layers[l-1](mods.hidstate))
        end
        if self.opts.recurrent then
            mods.outputs[#mods.outputs+1] = name
        end
    elseif name:sub(1, 10) == 'pre_action' then
        local nactions = self.opts.nactions_byname[name:sub(5)]
        assert(nactions > 0)
        if nactions == 1 then
            -- make continuous action
            mods[name] = nn.Linear(self.opts.hidsz, 2)(self:get('hidstate', mods))
        else
            mods[name] = nn.LogSoftMax()(
                nn.Linear(self.opts.hidsz, nactions)(self:get('hidstate', mods)))
        end
    elseif name:sub(1, 6) == 'action' then
        -- sample action
        local nactions = self.opts.nactions_byname[name]
        assert(nactions > 0)
        if nactions == 1 then
            mods[name] = nn.GaussianSampler()(self:get('pre_' .. name, mods))
        else
            mods[name] = nn.Sampler()(self:get('pre_' .. name, mods))
        end
        mods.outputs[#mods.outputs+1] = name
    elseif name == 'value' then
        mods.value = nn.Linear(self.opts.hidsz, 1)(self:get('hidstate', mods))
        mods.outputs[#mods.outputs+1] = name
    end
end

function Agent:get(name, mods)
    if not mods[name] then
        self:build(name, mods)
    end
    return mods[name]
end

function Agent:get_outputs(mods)
    for i = 1, #self.opts.action_names do
        self:get(self.opts.action_names[i], mods)
    end
    self:get('value', mods)
end

function Agent:build_model()
    local mods = {}
    mods.inputs = {}
    mods.outputs = {}
    self:get_outputs(mods)

    self.in_inds = {}
    self.out_inds = {}
    local in_mods = {}
    local out_mods = {}
    for i = 1, #mods.inputs do
        self.in_inds[mods.inputs[i]] = i
        in_mods[i] = mods[mods.inputs[i]]
    end
    for i = 1, #mods.outputs do
        self.out_inds[mods.outputs[i]] = i
        out_mods[i] = mods[mods.outputs[i]]
    end
    -- nngraph.annotateNodes()
    local model = nn.gModule(in_mods, out_mods)
    model.mods = mods
    return model
end

function Agent:zero_grads()
    for i = 1, #self.paramx do
        self.paramdx[i]:zero()
    end
end

function Agent:update_param()
    for i = 1, #self.paramx do
        local x = self.paramx[i]
        local dx = self.paramdx[i]
        if not self.optim_state[i] then self.optim_state[i] = {} end
        local state = self.optim_state[i]
        local f = function(x0) return x, dx end
        local config = {learningRate = self.opts.lrate}
        if self.opts.optim == 'sgd' then
            config.momentum = self.opts.momentum
            config.weightDecay = self.opts.wdecay
            optim.sgd(f, x, config, state)
        elseif self.opts.optim == 'rmsprop' then
            config.alpha = self.opts.rmsprop_alpha
            config.epsilon = self.opts.rmsprob_eps
            config.weightDecay = self.opts.wdecay
            optim.rmsprop(f, x, config, state)
        elseif self.opts.optim == 'adam' then
            config.beta1 = self.opts.adam_beta1
            config.beta2 = self.opts.adam_beta2
            config.epsilon = self.opts.adam_eps
            optim.adam(f, x, config, state)
        else
            error('wrong optim')
        end
    end
    if self.opts.encoder_lut then
        self:zero_encoder_nil()
    end
end

function Agent:get_input(name)
    if name == 'prev_hid' then
        if self.t > 1 then
            local prev_out = self.states[self.t - 1].out
            return prev_out[self.out_inds['hidstate']]
        else
            return torch.zeros(self.size, self.opts.hidsz)
        end
    end
end

function Agent:forward(obs, t, active)
    self.t = t
    self.states[t] = {}
    local state = self.states[t]
    state.input = {}
    for name, i in pairs(self.in_inds) do
        if name == 'obs' then
            state.input[i] = obs
        else
            state.input[i] = self:get_input(name)
        end
    end
    if #state.input == 1 then state.input = state.input[1] end
    state.out = self.model_clones[self.t]:forward(state.input)
    state.active = active
    local a = {}
    for i = 1, #self.opts.action_names do
        local name = self.opts.action_names[i]
        state[name] = state.out[self.out_inds[name]]
        table.insert(a, state[name])
    end
    if #a == 1 then a = a[1] end
    return a
end

function Agent:get_grad(name, state)
    if name:sub(1, 6) == 'action' then
        local value = state.out[self.out_inds['value']]:clone()
        value:cmul(state.active)
        local cost
        local a = self.opts.constant_baseline
        if a > 0 then -- use constant baseline instead of nn model
            assert(a < 1)
            if not self.simple_baseline then self.simple_baseline = 0 end
            self.simple_baseline =a*self.simple_baseline + (1-a)*state.value_target:mean()
            cost = state.value_target:clone():mul(-1):add(self.simple_baseline)
        else
            cost = -(state.value_target - value)
        end
        cost:cmul(state.active)
        cost:div(self.opts.batch_size)
        local mod = self.model_clones[self.t].mods[name].data.module
        mod.cost = cost
        mod.entropy_reg = torch.zeros(self.size)
        mod.entropy_reg:fill(self.opts.entropy_reg / self.opts.batch_size)
        mod.entropy_reg:cmul(state.active)
        if self.opts.nminds > 1 and self.opts.sp_test_entr_zero then
            for i = 1, self.size do
                if self.batch[math.ceil(i/self.opts.nagents)].type == 'test_task' then
                    mod.entropy_reg[i] = 0
                end
            end
        end
        return torch.zeros(self.size, self.opts.nactions_byname[name])
    elseif name == 'value' then
        local value = state.out[self.out_inds['value']]:clone()
        value:cmul(state.active)
        local cost = self.value_loss:forward(value, state.value_target)
        local g = self.value_loss:backward(value, state.value_target)
        g:mul(self.opts.alpha)
        self.stat.value_cost = (self.stat.value_cost or 0) + cost
        self.stat.value_count = (self.stat.value_count or 0) + state.active:sum()
        g:div(self.opts.batch_size)
        if self.opts.debug then
            print('B' .. self.t, 'value', value[1][1], state.value_target[1][1])
        end
        return g
    elseif name == 'hidstate' then
        local next_state = self.states[self.t + 1]
        if next_state and next_state.gradInputs then
            return next_state.gradInputs[self.in_inds['prev_hid']]
        else
            return torch.zeros(self.size, self.opts.hidsz)
        end
    end
end

function Agent:set_value_target(reward)
    local state = self.states[self.t]
    state.value_next = torch.zeros(self.size, 1)
    local next_state = self.states[self.t+1]
    if self.opts.mode == 'pg' then
        if next_state.reward_sum then
            state.reward_sum = next_state.reward_sum:clone()
        else
            state.reward_sum = torch.zeros(self.size, 1)
            state.reward_sum:copy(self.reward_terminal)
        end
        state.value_next = state.reward_sum:clone()
        state.reward_sum:add(reward)
    else
        state.value_next = next_state.out[self.out_inds['value']]:clone()
        state.value_next:cmul(next_state.active)
        if self.t == self.opts.max_steps then
            state.value_next:copy(self.reward_terminal)
        end
    end
    state.value_target = state.value_next:clone()
    state.value_target:add(reward)
    state.value_target:cmul(state.active)
end

function Agent:backward(t, reward)
    self.t = t
    self:set_value_target(reward)
    local state = self.states[t]
    local gradOutput = {}
    for name, i in pairs(self.out_inds) do
        local g = self:get_grad(name, state)
        local mask = state.active:view(-1, 1):clone()
        g:cmul(mask:expandAs(g):clone())
        gradOutput[i] = g
    end
    state.gradInputs = self.model_clones[t]:backward(state.input, gradOutput)
end

function Agent:show_stat()
end

function Agent:show_map()
end
