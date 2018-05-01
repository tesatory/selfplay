require('xlua')
paths.dofile('Worker.lua')

local function print_sorted_bykeys(t)
    local k = {}
    for i in pairs(t) do table.insert(k, i) end
    table.sort(k)
    for _, i in ipairs(k) do
        if type(t[i]) == 'table' then
        elseif type(t[i]) == 'userdata' then
        else
            print(i .. ' : ' ..  t[i])
        end
    end
end


local Trainer = torch.class('Trainer')

function Trainer:__init(opts)
    self.opts = opts
    self:init_workers()
    self.log = {}
end

function Trainer:init_workers()
    self.worker_local = Worker(self.opts)
    self.worker_local.is_local = true
    if self.opts.nworker > 1 then
        print('starting ' .. self.opts.nworker .. ' workers')
        local threads = require('threads')
        threads.Threads.serialization('threads.sharedserialize')
        local opts = self.opts
        self.workers = threads.Threads(self.opts.nworker,
            function(idx)
                if opts.rand_seed > 0 then
                    torch.manualSeed(opts.rand_seed + idx * 123456)
                end
                paths.dofile('Worker.lua')
            end)
        self.workers:specific(true)
        for w = 1, self.opts.nworker do
            self.workers:addjob(w,
                function(opts)
                    g_worker = Worker(opts)
                end,
                function() end, self.opts)
        end
        if self.opts.sc_count_alpha > 0 then
            -- TODO: ore x scv x barracks x depot x marines
            self.state_counts = self.state_counts or torch.zeros(41, 41, 41, 41, 20)
            for w = 1, self.opts.nworker do
                self.workers:addjob(w,
                    function(state_counts)
                        g_worker.state_counts = state_counts
                    end,
                    function() end, self.state_counts
                )
            end
        end
        self.workers:synchronize()
    end
end

function Trainer:set_hardness(h)
    self.worker_local.factory:set_hardness(h)
    if self.opts.nworker > 1 then
        for w = 1, self.opts.nworker do
            self.workers:addjob(w,
                function(hh)
                    g_worker.factory:set_hardness(hh)
                end,
                function() end, h)
        end
    end
end

function Trainer:train(N)
    for n = 1, N do
        local ep = #self.log + 1

        -- in self-play, can terminate after certain number of test steps
        if self.opts.sp_test_steps_max > 0 and ep > 1
        and self.log[ep-1].test_steps
        and self.log[ep-1].test_steps >= self.opts.sp_test_steps_max then
            break
        end

        local stat = {}
        if trainer.opts.entropy_reg_decay > 0 then
            trainer.opts.entropy_reg = trainer.opts.entropy_reg * trainer.opts.entropy_reg_decay
        end
        if self.opts.curriculum_end > 0 then
            -- adjust curriculum
            assert(self.opts.curriculum_end > self.opts.curriculum_sta)
            local h = (ep - self.opts.curriculum_sta)/(self.opts.curriculum_end - self.opts.curriculum_sta)
            h = math.min(1, math.max(0, h))
            self:set_hardness(h)
        end
        for k = 1, self.opts.nbatches do
            if self.opts.show then xlua.progress(k, self.opts.nbatches) end
            if self.opts.nworker > 1 then
                self.worker_local.agent:zero_grads()
                for w = 1, self.opts.nworker do
                    self.workers:addjob(w,
                        function(opts, paramx)
                            for k, v in pairs(opts) do
                                g_worker.opts[k] = v
                            end
                            for i = 1, #paramx do
                                g_worker.agent.paramx[i]:copy(paramx[i])
                            end
                            local stat = g_worker:run_episode()
                            collectgarbage("collect")
                            return g_worker.agent.paramdx, stat
                        end,
                        function(paramdx, s)
                            for i = 1, #paramdx do
                                self.worker_local.agent.paramdx[i]:add(paramdx[i])
                            end
                            merge_stat(stat, s)
                        end,
                        self.opts, self.worker_local.agent.paramx
                    )
                end
                self.workers:synchronize()
                for i = 1, #self.worker_local.agent.paramdx do
                    self.worker_local.agent.paramdx[i]:div(self.opts.nworker)
                end
                collectgarbage("collect")
            else
                local s = self.worker_local:run_episode()
                merge_stat(stat, s)
            end

            if self.opts.max_grad_norm > 0 then
                local grad_norm = 0
                for _, x in pairs(self.worker_local.agent.paramdx) do
                    grad_norm = grad_norm + torch.pow(x, 2):sum()
                end
                grad_norm = math.sqrt(grad_norm)
                stat.max_grad_norm = stat.max_grad_norm or 0
                stat.max_grad_norm = math.max(stat.max_grad_norm, grad_norm)
                if grad_norm > self.opts.max_grad_norm then
                    for _, x in pairs(self.worker_local.agent.paramdx) do
                        x:div(grad_norm / self.opts.max_grad_norm)
                    end
                end
            end

            self.worker_local.agent:update_param()
        end

        stat.param_norm = 0
        for _, x in pairs(self.worker_local.agent.paramdx) do
            stat.param_norm = stat.param_norm + torch.pow(x, 2):sum()
        end
        stat.param_norm = math.sqrt(stat.param_norm)

        for k, v in pairs(stat) do
            if string.sub(k, 1, 5) == 'count' then
                local s = string.sub(k, 6)
                stat['reward' .. s] = stat['reward' .. s] / v
                stat['success' .. s] = stat['success' .. s] / v
                if self.opts.rllab then
                    stat['reward' .. s] = stat['reward' .. s] / self.opts.rllab_reward_coeff
                end
            end
        end
        stat.value_cost = stat.value_cost / stat.value_count
        
        local sc_actions = {'action_move', 'action_mine', 'action_train',
            'action_build'}
        local sc_unit_counts, sc_switch_state
        if self.opts.sc then
            for _, a in pairs(sc_actions) do
                if stat[a] then
                    stat[a] = stat[a] / stat.value_count
                end
            end
            if stat.unit_counts then
                sc_unit_counts = {}
                for k, v in pairs(stat.unit_counts) do
                    stat.unit_counts[k] = v / stat.count
                    stat[k] = stat.unit_counts[k]
                    table.insert(sc_unit_counts, k)
                end
            end
            if self.opts.sc and stat.switch_state then
                sc_switch_state = {}
                for k, v in pairs(stat.switch_state) do
                    stat.switch_state[k] = v / stat.self_play_count
                    stat['switch_' .. k] = stat.switch_state[k]
                    table.insert(sc_switch_state, 'switch_' .. k)
                end
            end
        end

        if self.opts.nminds > 1 then
            if stat.self_play_count and stat.self_play_count > 0 then
                for m = 1, self.opts.nminds do
                    if stat['reward_mind' .. m] then
                        stat['reward_mind' .. m] = stat['reward_mind' .. m] / stat.self_play_count
                        if self.opts.rllab then
                            stat['reward_mind' .. m] = stat['reward_mind' .. m] / self.opts.rllab_reward_coeff
                        end
                    end
                end
                if stat.door_cross then
                    stat.door_cross = stat.door_cross / stat.self_play_count
                end
                if stat.switch_t then
                    stat.switch_t = stat.switch_t / stat.self_play_count
                end
                if stat.switch_dist then
                    stat.switch_dist = stat.switch_dist / stat.self_play_count
                end
                if stat.switch_count then
                    stat.switch_count = stat.switch_count / stat.self_play_count
                end
                if stat.lock then
                    -- stat.lock_dist = stat.lock_dist / stat.lock
                    stat.lock = stat.lock / stat.self_play_count
                end
                if stat.lamp then
                    -- stat.lamp_dist = stat.lamp_dist / stat.lamp
                    stat.lamp = stat.lamp / stat.self_play_count
                end
                if stat.toggle_count then
                    stat.toggle_count = stat.toggle_count / stat.self_play_count
                end
                if stat.return_t then
                    stat.return_t = stat.return_t / stat.self_play_count
                end
                if stat.push_dist then
                    stat.push_dist = stat.push_dist / stat.self_play_count
                end
                if stat.swamp then
                    stat.swamp = stat.swamp / stat.self_play_count
                end
            end
        end
        if stat.position then
            g_position_log = stat.position.data:clone()
        end

        if #self.log > 0 then
            if stat.test_steps or self.log[#self.log].test_steps then
                stat.test_steps = (stat.test_steps or 0) + (self.log[#self.log].test_steps or 0)
            end
        end

        stat.epoch = #self.log + 1
        -- print(format_stat(stat))
        -- print(stat)
        print_sorted_bykeys(stat)
        if stat.alice_actions then
            stat.alice_actions:div(stat.alice_actions:sum())
            m = ''
            for i = 1, stat.alice_actions:size(1) do
                m = m .. stat.alice_actions[i] .. ', '
            end
            print('alice_actions'  .. ' : ' ..  m)
        end

        table.insert(self.log, stat)

        if self.opts.plot then
            if stat.epoch > 1 then
                self:plot()
            end
        end

        if self.opts.save ~= '' and self.opts.save_all then
            self:save(self.opts.save .. '.' .. stat.epoch)
        end

    end
end

function Trainer:plot()
    local stat = self.log[#self.log]
    plot_stat(self.log, {'reward','reward_self_play', 'reward_test_task'})
    plot_stat(self.log, {'success','success_self_play', 'success_test_task'})
    plot_stat(self.log, 'value_cost')
    plot_stat(self.log, 'param_norm')
    -- plot_stat(self.log, 'max_grad_norm')
    if self.opts.sc then
        plot_stat(self.log, sc_actions)
        if sc_unit_counts then
            plot_stat(self.log, sc_unit_counts, 'units')
        end
    end
    if self.opts.nminds > 1 then
        local reward_mind = {}
        for m = 1, self.opts.nminds do
            table.insert(reward_mind, 'reward_mind' .. m)
        end
        plot_stat(self.log, reward_mind)
        plot_stat(self.log, {'switch_t', 'return_t'})
        if not self.opts.sc then
            plot_stat(self.log, 'switch_dist', 'push_dist')
        end
        if not self.opts.rllab and not self.opts.sc then
            plot_stat(self.log, {'door_cross', 'lock', 'lamp','switch_count', 'toggle_count'})
        end
        if stat.swamp then
            plot_stat(self.log, 'swamp')
        end
        if self.opts.sc and sc_switch_state then
            plot_stat(self.log, sc_switch_state, 'switch state')
        end
        plot_stat(self.log, 'test_steps')
        if self.log[#self.log].test_steps then
            plot_stat(self.log, 'reward_test_task', nil, 'test_steps')
        end
    end
    if g_position_log then
        plot_switch_pos(g_position_log, self.opts)
    end
end

function Trainer:test()
    self.worker_local.test_run = true
    self.worker_local:run_episode()
    self.worker_local.test_run = false
end

function Trainer:save(path)
    local f = {}
    f.paramx = self.worker_local.agent.paramx
    f.optim_state = self.worker_local.agent.optim_state
    f.log = self.log
    f.opts = self.opts
    f.state_counts = self.state_counts
    torch.save(path, f)
end

function Trainer:load(path)
    local f = torch.load(path)
    for i = 1, #f.paramx do
        self.worker_local.agent.paramx[i]:copy(f.paramx[i])
    end
    self.worker_local.agent.optim_state = f.optim_state
    self.log = f.log
    if f.state_counts then
        if self.state_counts then
            self.state_counts:copy(f.state_counts)
        else
            print('W: state count ignored')
        end
    end
end
