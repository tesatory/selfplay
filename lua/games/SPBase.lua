-- Base class for self-play games:
-- First, agent A controls the environment.
-- If A choose stop action, then control is transfered to agent B

local SPBase, parent = torch.class('SPBase', 'MazeBase')

function SPBase:__init(opts, vocab)
    parent.__init(self, opts, vocab)
    self.max_info = opts.max_info or 0
    self.return_loc_only = opts.return_loc_only
    self.test_rate = opts.test_rate
    self.test_mind_separate = opts.test_mind_separate
    self.rand_alice = (g_opts.sp_rand_alice == 1) -- TODO: bit hacky?
    self.sp_mode = g_opts.sp_mode
    self.reward_bob_step = g_opts.sp_reward_bob_step

    if torch.uniform() < self.test_rate then
        -- test task for agent
        self.test_mode = true
        self.type = 'test_task'
    else
        self.test_mode = false
        self.type = 'self_play'
    end

    self.stat = {}
    self:init_env()

    -- use STOP action for transfer control
    self.agent:add_action('stop',
        function(self)
            if not self.maze.switch_done then
                self.maze:switch_agent()
            end
        end)

    self.agent_init_loc_y = self.agent.loc.y
    self.agent_init_loc_x = self.agent.loc.x

    -- for random alice
    self.agent.act_orig = self.agent.act
    self.agent.rand_switch_t = torch.random(math.floor(g_opts.max_steps / 2))
    self.agent.act = function(self, action_id)
        if self.maze.rand_alice and self.maze.current_mind == 1 then
            if self.maze.t + 1 == self.rand_switch_t then
                action_id = self.action_ids['stop']
            else
                action_id = torch.random(self.nactions)
                while self.action_ids['stop'] == action_id do
                    action_id = torch.random(self.nactions)
                end
            end
        end
        self:act_orig(action_id)
    end

    self.finished = false
    self.current_mind = 1
    if self.test_mode then
        if self.test_mind_separate then
            self.current_mind = 3
        else
            self.current_mind = 2
        end
        self.stat.test_task_count = 1
    else
        self.stat.self_play_count = 1
    end

    if not self.return_loc_only then
        self.target_state = self:get_state_snapshot()
    end

end

function SPBase:init_env()
    error('need override')
end

function SPBase:switch_agent()
    self.switch_done = true
    self.current_mind = 2
    self.switch_t = self.t + 1
    self.agent.attr._ascii_color = {'blue', 'reverse'}
    self.stat.switch_t = self.switch_t
    self.stat.switch_dist = math.abs(self.agent.loc.y - self.agent_init_loc_y) +
        math.abs(self.agent.loc.x - self.agent_init_loc_x)
    if self.sp_mode == 'reverse' then
    elseif self.sp_mode == 'repeat' then
        self.target_state = self:get_state_snapshot()
        self:reset()
    elseif self.sp_mode == 'compete' then
        -- randomly decide
        if torch.uniform() < 0.5 then
            self.current_mind = 1
        end
        -- let Alice know she has to perform task
        self.agent.attr.sp_mode = 'perform_task'
    else
        error('wrong mode')
    end
end

function SPBase:reset()
    self.map:remove_item(self.agent)
    self.agent.loc.y = self.agent_init_loc_y
    self.agent.loc.x = self.agent_init_loc_x
    self.map:add_item(self.agent)
end

function SPBase:try_finish_game()
    if self:is_success() then
        self.finished = true
    end
end

function SPBase:update()
    parent.update(self)
    if self.switch_done then
        self:try_finish_game()
    end
    self.agent.attr.time = 'time' .. self.t
end

function SPBase:get_terminal_reward_test()
    return 0
end

function SPBase:get_state_snapshot()
    local visibility = math.max(self.map.height, self.map.width) - 1
    local indim = (2*visibility+1)^2 + self.max_info
    local data = torch.zeros(indim, self.nwords)
    local tmp_time = self.agent.attr.time
    self.agent.attr.time = nil
    self:get_visible_state(data, false, visibility)
    self.agent.attr.time = tmp_time
    return data
end

function SPBase:is_success()
    local success = false
    if self.test_mode or self.return_loc_only then
        if self.goal.loc.y == self.agent.loc.y and self.goal.loc.x == self.agent.loc.x then
            success = true
        end
    else
        -- full stats must match
        local x = self.agent.attr.sp_mode -- TODO
        self.agent.attr.sp_mode = nil
        local current_state = self:get_state_snapshot()
        if self.switch_done and current_state:ne(self.target_state):sum() == 0 then
            success = true
        end
        self.agent.attr.sp_mode = x
    end
    return success
end

function SPBase:get_reward()
    if not self.test_mode and self.current_mind == 2 and self.reward_bob_step then
        return -0.1
    else
        return 0
    end
end

function SPBase:get_terminal_reward_mind(mind)
    if self.test_mode then
        if mind == self.current_mind then
            return self:get_terminal_reward_test()
        else
            return 0
        end
    end

    local success = self:is_success()

    if mind == 1 then
        if self.sp_mode == 'compete' then
            if self.switch_done then
                if self.current_mind == 2 then
                    return (self.t - self.switch_t) * 0.1
                else
                    return -(self.t - self.switch_t) * 0.1
                end
            else
                return 0
            end
        else
            if self.current_mind == 2 then
                return math.max(0, (self.t - self.switch_t * 2) * 0.1)
            else
                return 0
            end
        end
    elseif mind == 2 and not self.reward_bob_step then
        if self.current_mind == 2 then
            return - (self.t -self.switch_t) * 0.1
        else
            return 0
        end
    else
        return 0
    end
end
