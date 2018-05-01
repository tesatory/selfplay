local SCSPEconEnv, parent = torch.class('SCSPEconEnv', 'SCEconEnv')

function SCSPEconEnv:__init(opts, torchcraft, vocab)
    parent.__init(self, opts, torchcraft, vocab)
    self.stat.switch_t = 0
    self.stat['switch_count'] = 0
    if torch.uniform() < self.opts.sp_test_rate then
        self.test_mode = true
        self.current_mind = 2
        self.type = 'test_task'
        self.stat['test_task_count'] = 1
    else
        self.test_mode = false
        self.current_mind = 1
        self.type = 'self_play'
        self.stat['self_play_count'] = 1
    end
end

function SCSPEconEnv:observe_info(obs)
    parent.observe_info(self, obs)

    if self.test_mode then
        self:observe_word(obs, 'test_mode', obs.info_loc)
    else
        self:observe_word(obs, 'self_play', obs.info_loc)
    end
    obs.info_loc = obs.info_loc + 1

    -- time
    local n = math.floor(self.opts.vocab_num_limit * self.t / self.opts.max_steps)
    self:observe_word(obs, 'num' .. n, obs.info_loc)
    obs.info_loc = obs.info_loc + 1

    if not self.test_mode and self.current_mind == 2 then
        -- target state for Bob
        self:observe_state(self.stat.switch_state, obs)
    end
end

function SCSPEconEnv:act(agent_ind, action)
    local agent_uid = self.agent_ind2uid[agent_ind]
    local agent = self.agents[agent_ind]

    if self.current_mind == 1 and agent.type == self.tc.unittypes.Terran_Command_Center and action == 2 then
        self:switch_mind()
    else
        parent.act(self, agent_ind, action)
    end
end

function SCSPEconEnv:switch_mind()
    self.current_mind = 2
    self.stat.switch_t = self.t + 1
    self.stat['switch_count'] = 1
    self.stat.switch_state = self:get_state_snapshot()

    -- restart SC
    for uid, ut in pairs(self.tc.state.units_myself) do
        if ut.type == self.tc.unittypes.Zerg_Infested_Terran then
            table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit,
                uid, self.tc.cmd.Move, 0, 22, 33))
            self.sp_restarting = true
            break
        end
    end
end

function SCSPEconEnv:is_bob_success()
    local state = self:get_state_snapshot()
    for k, v in pairs(self.stat.switch_state) do
        if v > state[k] then
            return false
        end
    end
    return true
end

function SCSPEconEnv:restart_compete()
    -- finish restart proccess
    while self.tc.state.battle_just_ended == false do
        if self.tc.DEBUG > 1 then print('sp restart wait') end
        self.tc:send({''})
        local update = self.tc:receive()
    end
    while self.tc.state.waiting_for_restart do
        if self.tc.DEBUG > 1 then print('waiting_for_restart') end
        self.tc:send({''})
        local update = self.tc:receive()
    end
    self.state = self.tc.state
    -- unit IDs must've changed
    self.agent_ind2uid = {}
    self.agent_uid2ind = {}
    self:update_state()
    self.sp_restarting = false
end

function SCSPEconEnv:update()
    parent.update(self)
    if not self.test_mode then
        if self.sp_restarting then
            self:restart_compete()
        elseif self.current_mind == 2 and self:is_bob_success() then
            self.finished = true
        end
        if self.current_mind == 2 then
            self.stat.return_t = self.t - self.stat.switch_t
        end
    end
end

function SCSPEconEnv:get_reward(agent_ind)
    if self.test_mode then
        return parent.get_reward(self, agent_ind)
    else
        return 0
    end
end

function SCSPEconEnv:get_terminal_reward(agent_ind)
    if self.test_mode then
        return parent.get_terminal_reward(self, agent_ind)
    else
        return 0
    end
end

function SCSPEconEnv:get_terminal_reward_mind(mind)
    if self.test_mode then
        return 0
    end

    if mind == 1 then
        if self.current_mind == 2 then
            return self.opts.sp_reward_coeff * math.max(0, (self.t - 2 * self.stat.switch_t))
        else
            return 0
        end
    else
        return self.opts.sp_reward_coeff * (self.stat.switch_t - self.t)
    end
end

function SCSPEconEnv:show_stat()
    local state = self:get_state_snapshot()
    print(self.t, self.current_mind)
    print(state)
end