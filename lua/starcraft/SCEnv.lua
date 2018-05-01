local SCEnv = torch.class('SCEnv')

function SCEnv:__init(opts, torchcraft, vocab)
    self.opts = opts
    self.vocab = vocab
    self.tc = torchcraft
    if self.tc.DEBUG > 1 then print('init receive') end
    local update = self.tc:receive()
    self.state = self.tc.state
    if self.tc.DEBUG > 1 then
        print('init: ',
            self.state.battle_just_ended,
            self.state.waiting_for_restart,
            self.state.game_ended)
    end
    while self.state.waiting_for_restart do
        if self.tc.DEBUG > 1 then print('waiting_for_restart') end
        self.tc:send({''})
        update = self.tc:receive()
        self.state = self.tc.state
    end

    self.t = 0
    self.finished = false

    -- map between starcraft unit ID and agent index for training
    self.agent_ind2uid = {}
    self.agent_uid2ind = {}

    self.stat = {}
    self:update_state()
    self.actions_buffer = {}
end

function SCEnv:get_agents()
    return self.state.units_myself
end

function SCEnv:word2ind(w)
    if self.vocab[w] == nil then
        error('word not in vocab: ' .. w)
    end
    return self.vocab[w]
end

function SCEnv:observe_word(obs, word, loc_ind)
    if self.opts.encoder_lut then
        assert(obs.offset <= obs.data:size(1), 'increase encoder_lut_size!')
        obs.data[obs.offset] = (loc_ind-1)*self.opts.nwords + self:word2ind(word)
        obs.offset = obs.offset + 1
    else
        obs.data[loc_ind][self:word2ind(word)] = 1
    end
end

function SCEnv:observe_unit(obs, team, ut, loc_ind)
    self:observe_word(obs, team, loc_ind)
    self:observe_word(obs, self.tc.const.unittypes[ut.type], loc_ind)
end

function SCEnv:observe_info(obs)
end

function SCEnv:observe(agent_ind, data, visibility)
    if self.finished then return end
    local agent = self.agents[agent_ind]
    if agent == nil then return end
    local y = agent.position[2]
    local x = agent.position[1]
    local obs = {data = data}
    if self.opts.encoder_lut then obs.offset = 1 end
    local teams = {team_myself = self.tc.state.units_myself,
        team_neutral = self.tc.state.units_neutral,
        team_enemy = self.tc.state.units_enemy}
    for team, units in pairs(teams) do
        for uid, ut in pairs(units) do
            local dy = ut.position[2] - y
            local dx = ut.position[1] - x
            dy = math.floor(dy / self.opts.sc_resolution + 0.5)
            dx = math.floor(dx / self.opts.sc_resolution + 0.5)
            if math.abs(dx) <= visibility and math.abs(dy) <=visibility then
                local data_y = dy + visibility + 1
                local data_x = dx + visibility + 1
                local loc_ind = (data_y-1)*(2*visibility+1) + data_x
                self:observe_unit(obs, team, ut, loc_ind)
            end
        end
    end

    obs.info_loc = (2*visibility+1)^2 + 1
    self:observe_info(obs)
end

function SCEnv:agents_map_sync()
    -- keep agent list synced
    for uid, ut in pairs(self:get_agents()) do
        if not self.agent_uid2ind[uid] then
            self.agent_ind2uid[#self.agent_ind2uid+1] = uid
            self.agent_uid2ind[uid] = #self.agent_ind2uid
        end
    end
    assert(#self.agent_ind2uid <= self.opts.nagents)
end

function SCEnv:update_state()
    -- called after every recieve from SC
    self:agents_map_sync()
    self.agents = {}
    for i, uid in pairs(self.agent_ind2uid) do
        self.agents[i] = self.state.units_myself[uid]
    end
end

function SCEnv:is_active(agent_ind)
    if self.finished then
        return false
    elseif self.agents[agent_ind] == nil then
        return false
    else
        return true
    end
end

function SCEnv:update()
    local actions = table.concat(self.actions_buffer, ':')
    if self.tc.DEBUG > 1 then print('sending action:', actions) end
    self.tc:send({actions})
    self.actions_buffer = {}

    self.t = self.t + 1

    if self.tc.DEBUG > 1 then print('update receive') end
    local update = self.tc:receive()
    self.state = self.tc.state
    if self.tc.DEBUG > 1 then
        print('update: ',
            self.state.battle_just_ended,
            self.state.waiting_for_restart,
            self.state.game_ended)
    end
    self:update_state()

    if self.state.battle_just_ended or self.state.game_ended then
        if self.tc.DEBUG > 1 then print('finished') end
        self.finished = true
        self.tc.restart_needed = false
    end
end

function SCEnv:act(agent_ind, action)
    local agent_uid = self.agent_ind2uid[agent_ind]
    local agent = self.agents[agent_ind]
    local x = agent.position[1]
    local y = agent.position[2]

    local offset =32


    -- only move around
    if action == 1 then
        table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit_protected,
            agent_uid, self.tc.cmd.Move, -1, x + offset, y, -1))
    elseif action == 2 then
        table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit_protected,
            agent_uid, self.tc.cmd.Move, -1, x - offset, y, -1))
    elseif action == 3 then
        table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit_protected,
            agent_uid, self.tc.cmd.Move, -1, x, y + offset, -1))
    elseif action == 4 then
        table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit_protected,
            agent_uid, self.tc.cmd.Move, -1, x, y -offset, -1))
    else
        error('wrong action')
    end
end

function SCEnv:get_reward(agent_ind)
    return 0
end

function SCEnv:get_terminal_reward(agent_ind)
    return 0
end
