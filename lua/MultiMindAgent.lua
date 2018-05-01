paths.dofile('mods/SplitSwitch.lua')
paths.dofile('mods/SelectSwitch.lua')

local MultiMindAgent, parent = torch.class('MultiMindAgent', 'Agent')

function MultiMindAgent:__init(opts)
    self.nminds = opts.nminds
    parent.__init(self, opts)
end

function MultiMindAgent:build(name, mods)
    if name == 'mind' or name == 'target' then
        -- inputs to the model
        mods[name] = nn.Identity()()
        mods.inputs[#mods.inputs+1] = name
    elseif name == 'pre_hid' then
        local enc, lut = self:build_encoder(self.opts.hidsz)
        if lut then mods['encoder_lut'] = lut end
        mods.enc = enc
        mods.pre_hid = enc(self:get('obs', mods))
        if self.opts.mind_target then
            local target_enc, target_lut = self:build_encoder(self.opts.hidsz)
            if target_lut then mods['target_encoder_lut'] = target_lut end
            mods.target_enc = target_enc
            mods.pre_hid = nn.JoinTable(2)({mods.pre_hid, target_enc(self:get('target', mods))})
        end
    elseif name == 'hidstate' then
        -- internal hidden state
        mods.hidstate = self:nonlin()(self:get('pre_hid', mods))
        for l = 2, self.opts.nlayers do
            mods.linear_layers = mods.linear_layers or {}
            if l == 2 and self.opts.mind_target then
                mods.linear_layers[l-1] = nn.Linear(self.opts.hidsz * 2, self.opts.hidsz)
            else
                mods.linear_layers[l-1] = nn.Linear(self.opts.hidsz, self.opts.hidsz)
            end
            mods.hidstate = self:nonlin()(mods.linear_layers[l-1](mods.hidstate))
        end
    elseif name:sub(1, 6) == 'action' then
        -- add sampler after
        mods[name] = nn.LogSoftMax()(
            nn.Linear(self.opts.hidsz, self.opts.nactions_byname[name])(self:get('hidstate', mods))
            )
        mods.outputs[#mods.outputs+1] = name
    else
        parent.build(self, name, mods)
    end
end

function MultiMindAgent:zero_encoder_nil()
    for i = 1, self.nminds do
        local mods = self.model.mods.mind_mods[i]
        mods['encoder_lut'].weight[self.encoder_lut_nil]:zero()
        if self.opts.mind_target then
            mods['target_encoder_lut'].weight[self.encoder_lut_nil]:zero()
        end
    end
end

function MultiMindAgent:build_model()
    local mods = {}
    mods.inputs = {}
    mods.mind_mods = {}
    local value_mods = {self:get('mind', mods)}
    local action_mods = {}
    for name, _ in pairs(self.opts.nactions_byname) do
        action_mods[name] = {self:get('mind', mods)}
    end
    for m = 1, self.nminds do
        mods.mind_mods[m] = {}
        mods.mind_mods[m].obs = self:get('obs', mods)
        mods.mind_mods[m].inputs = {'obs'}
        if self.opts.mind_target then
            mods.mind_mods[m].target = self:get('target', mods)
            mods.mind_mods[m].inputs = {'obs', 'target'}
        end
        mods.mind_mods[m].outputs = {}
        self:get_outputs(mods.mind_mods[m])
        table.insert(value_mods, mods.mind_mods[m].value)
        for name, _ in pairs(self.opts.nactions_byname) do
            table.insert(action_mods[name], mods.mind_mods[m][name])
        end
    end

    if self.opts.mind_target then
        for m = 1, self.nminds do
    	    mods.mind_mods[m].target_enc:share(mods.mind_mods[m].enc, 'weight','bias')
        end
    end

    -- share encoder
    if self.opts.minds_share_enc then
        for m = 2, self.nminds do
            mods.mind_mods[m].enc:share(mods.mind_mods[1].enc, 'weight','bias')
            -- mods.mind_mods[m].target_enc:share(mods.mind_mods[1].target_enc, 'weight','bias')
            for l = 2, self.opts.nlayers do
                mods.mind_mods[m].linear_layers[l-1]:share(mods.mind_mods[1].linear_layers[l-1], 'weight','bias')
            end

        end
    end

    mods.value = nn.SelectSwitch()(value_mods)
    mods.outputs = {'value'}
    for i = 1, #self.opts.action_names do
        local name = self.opts.action_names[i]
        mods[name .. '_prob'] = nn.SelectSwitch()(action_mods[name])
        mods[name] = nn.Sampler()(mods[name .. '_prob'])
        table.insert(mods.outputs, name)
    end

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

function MultiMindAgent:get_input(name)
    if name == 'mind' then
        local mind = torch.LongTensor(self.size)
        for i = 1, self.size do
            local g = self.batch[math.ceil(i/self.opts.nagents)]
            mind[i] = g.current_mind
            if self.opts.nminds > 2 then
                -- TODO temp multiple alice
                g.current_alice = g.current_alice or torch.random(self.opts.nminds - 1)
                if g.current_mind == 1 then
                    mind[i] = g.current_alice
                else
                    mind[i] = self.opts.nminds
                end
            end
        end
        assert(mind:max() <= self.nminds)
        self.states[self.t].mind = mind
        return mind
    elseif name == 'target' then
        assert(self.opts.nagents == 1)
        local target = torch.zeros(self.init_obs:size())
        if self.opts.encoder_lut then
            target:fill(self.encoder_lut_nil)
        end
        for i, g in pairs(self.batch) do
            if g.type == 'test_task' and (not g.encode_init_test) then
                assert(self.opts.encoder_lut)
                target[i][1] = self.batch[1].vocab['test_mode'] -- TODO temp
            elseif g.sp_mode == 'reverse' then
                target[i] = self.init_obs[i]
            elseif g.sp_mode == 'repeat' then
                if g.current_mind == 1 then
                    target[i] = self.init_obs[i]
                else
                    target[i] = self.switch_obs[i]
                end
            elseif g.sp_mode == 'compete' then
                target[i] = self.init_obs[i]
            else
                error('wrong mode')
            end
        end
        return target
    else
        return parent.get_input(self, name)
    end
end

function MultiMindAgent:forward(obs, t, active)
    if self.opts.mind_target then
        if t == 1 then
            self.init_obs = obs:clone()
            self.switch_obs = obs:clone()
        end
        -- take snapshot
        for i, g in pairs(self.batch) do
            if g.current_mind == 1 then
                self.switch_obs[i] = obs[i]
            end
        end
    end
    return parent.forward(self, obs, t, active)
end

function MultiMindAgent:set_value_target(reward)
    if not self.opts.mind_reward_separate then
        parent.set_value_target(self, reward)
        return
    end


    local state = self.states[self.t]
    local next_state = self.states[self.t+1]

    if next_state.reward_sum then
        state.reward_sum = next_state.reward_sum:clone()

        if self.opts.mode == 'ac' and (self.t % self.opts.ac_freq == 0) then
            for i = 1, self.size do
                local m = state.mind[i]
                if self.states[1].mind[i] == 1 and m == 2 and m == next_state.mind[i] and state.active[i] == 1 and next_state.active[i] == 1 then
                    state.reward_sum[i][m] = next_state.out[self.out_inds['value']][i][1]
                end
            end
        end
    else
        state.reward_sum = torch.zeros(self.size, self.nminds)
        state.reward_sum:copy(self.reward_terminal:view(-1,1):expandAs(state.reward_sum))
        state.reward_sum = state.reward_sum:view(#self.batch, self.opts.nagents, self.nminds)
        -- in multi-agent games, all agents get same reward
        for i = 1, #self.batch do
            if self.batch[i].get_terminal_reward_mind then
                for a = 1, self.opts.nagents do
                    for m = 1, self.nminds do
                        if self.opts.nminds > 2 then
                            -- TODO temp multiple alice
                            if m == self.batch[i].current_alice then
                                state.reward_sum[i][a][m] = state.reward_sum[i][a][m] + self.batch[i]:get_terminal_reward_mind(1)
                            elseif m == self.opts.nminds then
                                state.reward_sum[i][a][m] = state.reward_sum[i][a][m] + self.batch[i]:get_terminal_reward_mind(2)
                            end
                        else
                            state.reward_sum[i][a][m] = state.reward_sum[i][a][m] + self.batch[i]:get_terminal_reward_mind(m)
                        end
                        if self.batch[i].type == 'self_play' then
                            self.stat['reward_mind' .. m] = self.stat['reward_mind' .. m] or 0
                            self.stat['reward_mind' .. m] = self.stat['reward_mind' .. m] + state.reward_sum[i][a][m]
                        end
                    end
                end
            end
        end
        state.reward_sum = state.reward_sum:view(-1, self.nminds)
    end

    state.value_target = torch.zeros(self.size, 1)
    for i = 1, self.size do
        local m = state.mind[i]
        state.reward_sum[i][m] = state.reward_sum[i][m] + reward[i]
        state.value_target[i][1] = state.reward_sum[i][m]
        if self.batch[math.ceil(i/self.opts.nagents)].type == 'self_play' then
            self.stat['reward_mind' .. m] = self.stat['reward_mind' .. m] or 0
            self.stat['reward_mind' .. m] = self.stat['reward_mind' .. m] + reward[i]
        end
    end
    state.value_target:cmul(state.active)

end

function MultiMindAgent:show_stat()
    parent.show_stat(self)
    print('mind: ' .. self.states[self.t].mind[1])
end

function MultiMindAgent:show_stat_final()
    if self.batch[1].get_terminal_reward_mind then
        for m = 1, self.nminds do
            if self.opts.nminds > 2 then
                -- TODO temp multiple alice
                if m == self.batch[1].current_alice then
                    print('terminal reward mind', m, self.batch[1]:get_terminal_reward_mind(1))
                elseif m == self.opts.nminds then
                    print('terminal reward mind', m, self.batch[1]:get_terminal_reward_mind(2))
                end
            else
                print('terminal reward mind', m, self.batch[1]:get_terminal_reward_mind(m))
            end
        end
    end
end
