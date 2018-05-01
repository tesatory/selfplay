paths.dofile('Agent.lua')
paths.dofile('MultiMindAgent.lua')
paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')
local Worker = torch.class('Worker')

function Worker:__init(opts)
    self.opts = opts
    if self.opts.rllab then
        paths.dofile('rllab/RllabBridge.lua')
        self.env_bridge = RllabBridge(self.opts)
    elseif self.opts.sc then
        paths.dofile('starcraft/SCBridge.lua')
        self.env_bridge = SCBridge(self.opts)
        self.opts.nwords = self.env_bridge.nwords
    else
        local game_util = paths.dofile('games/init.lua')
        self.vocab, self.ivocab = game_util.init_vocab()
        self.opts.nwords = #self.ivocab
        local factory = game_util.init_factory(opts, self.vocab)
        self.env_bridge = MazeBridge(factory, self.opts)
    end
    if self.opts.nminds > 1 then
        self.agent = MultiMindAgent(self.opts)
    else
        self.agent = Agent(self.opts)
    end
    self.test_run = false
end

function Worker:forward(batch, t)
    local obs, active, reward
    obs = self.env_bridge:batch_input(batch)
    active = self.env_bridge:batch_active(batch)
    local action = self.agent:forward(obs, t, active)
    if t > self.opts.max_steps then return end
    self:show_map(batch, active, action)
    self.env_bridge:batch_act(batch, action, active)
    self.env_bridge:batch_update(batch, active)
    reward = self.env_bridge:batch_reward(batch, active)
    self:show_stat(batch, active, action, reward)
    return reward, obs
end

-- run an episode on mini-batch
function Worker:run_episode()
    local batch = self.env_bridge:batch_init(self.opts.batch_size)
    -- TODO: bit hacky?
    if self.state_counts then
        for i = 1, self.opts.batch_size do
            batch[i].state_counts = self.state_counts
        end
    end

    -- reset agent
    self.agent.size = self.opts.batch_size * self.opts.nagents
    self.agent.stat = {}
    self.agent.states = {}
    self.agent:zero_grads()
    self.agent.batch = batch -- TODO: bit hacky?

    local rewards = {}
    local reward_sum = torch.zeros(self.agent.size)
    local obs
    for t = 1, self.opts.max_steps + 1 do
        rewards[t], obs = self:forward(batch, t)
        if t <= self.opts.max_steps then
            reward_sum:add(rewards[t])
            if self.agent.states[t].reward_internal then
                rewards[t]:add(self.agent.states[t].reward_internal)
            end
        end
    end
    local reward_terminal = self.env_bridge:batch_terminal_reward(batch)
    reward_sum:add(reward_terminal)
    self.agent.reward_terminal = reward_terminal

    local success = self.env_bridge:batch_success(batch)
    if self.test_run then
        if not self.opts.rllab and not self.opts.sc then
            batch[1].map:print_ascii()
        end
        print('terminal reward: ' .. reward_terminal[1])
        print('total reward: ' .. reward_sum[1])
        print('success: ' .. success[1])
        if self.agent.show_stat_final then
            self.agent:show_stat_final()
        end
        return
    end

    -- do back-propagation
    for t = self.opts.max_steps, 1, -1 do
        self.agent:backward(t, rewards[t])
    end

    local stat = {}
    reward_sum = reward_sum:view(self.opts.batch_size, self.opts.nagents)
    if self.opts.coop then
        -- all agents must've same reward
        stat.reward = reward_sum:select(2,1):sum()
    else
        stat.reward = reward_sum:sum()
    end
    stat.success = success:sum()
    stat.count = self.opts.batch_size

    for i, g in pairs(batch) do
        local t = batch[i].type
        if t then
            if self.opts.coop then
                stat['reward_' .. t] = (stat['reward_' .. t] or 0) + reward_sum[i][1]
            else
                stat['reward_' .. t] = (stat['reward_' .. t] or 0) + reward_sum[i]:sum()
            end
            stat['success_' .. t] = (stat['success_' .. t] or 0) + success[i]
            stat['count_' .. t] = (stat['count_' .. t] or 0) + 1
        end
        if batch[i].stat then
            merge_stat(stat, g.stat)
        end
    end

    for k,v in pairs(self.agent.stat) do
        stat[k] = v
    end
    return stat
end

function Worker:test_action(batch, active, action)
    for i = 1, self.opts.nagents do
        if active[i] == 1 then
            local agent = get_agent(batch[1], i)
            if self.opts.hand then
                print('input action for agent', i)
                local k = io.read()
                if k == 'a' then
                    action[i][1] = agent.action_ids['left']
                elseif k == 'd' then
                    action[i][1] = agent.action_ids['right']
                elseif k == 'w' then
                    action[i][1] = agent.action_ids['up']
                elseif k == 's' then
                    action[i][1] = agent.action_ids['down']
                elseif k == 'z' then
                    action[i][1] = agent.action_ids['stop']
                elseif k == 't' then
                    action[i][1] = agent.action_ids['toggle']
                elseif k == 'A' then
                    action[i][1] = agent.action_ids['push_left']
                elseif k == 'D' then
                    action[i][1] = agent.action_ids['push_right']
                elseif k == 'W' then
                    action[i][1] = agent.action_ids['push_up']
                elseif k == 'S' then
                    action[i][1] = agent.action_ids['push_down']
                elseif k == 'g' then
                    action[i][1] = agent.action_ids['grab']
                elseif k == 'c' then
                    action[i][1] = agent.action_ids['craft']
                elseif k == 'x' then
                    action[i][1] = agent.action_ids['term']
                elseif k == '1' then
                    action[i][1] = agent.action_ids['task1']
                elseif k == '2' then
                    action[i][1] = agent.action_ids['task2']
                elseif k == '3' then
                    action[i][1] = agent.action_ids['task3']
                elseif k == '4' then
                    action[i][1] = agent.action_ids['task4']
                elseif k == '5' then
                    action[i][1] = agent.action_ids['task5']
                elseif k == '6' then
                    action[i][1] = agent.action_ids['task6']
                elseif k == '7' then
                    action[i][1] = agent.action_ids['task7']
                end
            end
        end
    end
end

function Worker:show_map(batch, active, action)
    if self.test_run then
        if active[1] == 1 then
            if self.opts.rllab then
                if self.opts.plot then
                    self.env_bridge:render()
                end
                if self.opts.rllab_save_image ~= '' then
                    self.env_bridge:save_image(batch[1], self.opts.rllab_save_image)
                end
            elseif self.opts.sc then
                --
                os.execute('sleep 0.02')
            else
                if self.opts.plot then
                    local agent = get_agent(batch[1], 1)
                    local aname = agent.action_names[action[1][1]]
                    local img = batch[1].map:to_image()
                    g_plot:image{img = img, win = 'game', options = {title = aname}}
                    self.agent:show_map()
                end
                batch[1].map:print_ascii()
                self:test_action(batch, active, action)
                os.execute('sleep 0.1')
            end
        end
    end
end

function Worker:show_stat(batch, active, action, reward)
    if self.test_run then
        if self.opts.rllab then
            if active[1] == 1 then
                print(self.agent.t, 'reward: ' .. reward[1])
                self.agent:show_stat()
            end
        elseif self.opts.sc then
            if active:narrow(1,1,self.opts.nagents):sum() > 0 then
                if batch[1].show_stat then
                    batch[1]:show_stat()
                end
            end
        else
            for a = 1, self.opts.nagents do
                if active[a] == 1 then
                    local agent = get_agent(batch[1], a)
                    print('agent ' .. a .. ' reward:', reward[a],
                        'action: ' .. agent.action_names[action[a][1]])
                    self.agent:show_stat()
                    if batch[1].show_stat then batch[1]:show_stat() end
                end
            end
        end
    end
end
