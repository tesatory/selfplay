-- Test task: agent has to turn on light, unlock the door, go through it to
-- reach the goal
--
-- Self play: agent A can turn off light, or lock the door to make it harder for
-- agent B to navigate back to the initial location


local SPLightKey, parent = torch.class('SPLightKey', 'SPBase')

function SPLightKey:__init(opts, vocab)
    self.with_wall = opts.with_wall
    self.with_switch = opts.with_switch
    self.with_lamp = opts.with_lamp
    self.lamp_off_prob = opts.lamp_off_prob
    self.push_block_door = opts.push_block_door
    self.nswamp = opts.nswamp
    self.swamp_stuck_prob = opts.swamp_stuck_prob
    self.test_rand_init = opts.test_rand_init
    parent.__init(self, opts, vocab)
    self.stat.alice_actions = torch.zeros(3) -- 1=lock 2=light 3=door
end

function SPLightKey:init_env()
    if self.with_wall then
        self:build_wall()
    end

    if self.test_mode then
        self.goal = self:place_item_rand({type = 'goal'})
    end

    if self.with_switch then
        assert(self.with_wall)
        assert(not self.push_block_door)
        local sy, sx = self.map:get_empty_loc()
        if self.test_rand_init == false and self.test_mode then
            sy, sx = self:rand_loc_opposite_to(self.goal)
        end
        self.switch = self:place_item({type = 'switch', controls = 'door',
            _c = torch.random(2), _cn = 2}, sy, sx)
        if self.test_rand_init == false and self.test_mode then
            -- lock the door
            self.switch.attr._c = 1
        end
        self.switch.attr.color = 'color' .. self.switch.attr._c
        self.switch_init_color = self.switch.attr._c
    end

    if self.with_lamp then
        assert(self.with_wall)
        local sy, sx = self.map:get_empty_loc()
        if self.test_rand_init == false and self.test_mode then
            sy, sx = self:rand_loc_opposite_to(self.goal)
        end
        self.lamp = self:place_item({type = 'switch', controls = 'lamp',
            _c = 2, _cn = 2}, sy, sx)
        if self.test_rand_init == false and self.test_mode then
            -- turn off
            self.lamp.attr._c = 1
        elseif torch.uniform() < self.lamp_off_prob then
            self.lamp.attr._c = 1                
        end
        self.lamp.attr.color = 'color' .. self.lamp.attr._c
        self.lamp_init_color = self.lamp.attr._c
    end

    assert(self.nagents == 1)
    self.agents = {}
    local ay, ax
    if self.test_mode then
        if self.test_rand_init == false and self.with_wall then
            ay, ax = self:rand_loc_opposite_to(self.goal)
        else
            ay, ax = self.map:get_empty_loc()
        end
    else
        ay, ax = self.map:get_empty_loc()
    end
    self.agents[1] = self:place_item({type = 'agent',
        name = 'agent1'}, ay, ax)
    self.agent = self.agents[1]
    if self.push_block_door then
        self.agent:add_push_actions()
    end

    self:add_default_items()

    if self.nswamp > 0 then
        for i = 1, self.nswamp do
            self:place_item_rand({type = 'swamp', _stuck_prob = self.swamp_stuck_prob})
        end
        for _, a in pairs({'up', 'down', 'left', 'right'}) do
            local id = self.agent.action_ids[a]
            local f = self.agent.actions[id]
            self.agent.actions[id] = function(self)
                local l = self.map.items[self.loc.y][self.loc.x]
                for _, e in ipairs(l) do
                    if e.type == 'swamp' then
                        if torch.uniform() < e.attr._stuck_prob then
                            -- got stuck, do nothing
                            if self.maze.current_mind == 1  then
                                self.maze.stat.swamp = 1
                            end
                            return
                        end
                    end
                end
                f(self)
            end            
        end
    end

    -- count toggle actions
    local toggle_id = self.agent.action_ids['toggle']
    local toggle_f = self.agent.actions[toggle_id]
    self.agent.actions[toggle_id] = function(self)
        if self.maze.current_mind == 1 then
            if self.maze.switch and self.maze.switch.loc.y == self.loc.y 
                and self.maze.switch.loc.x == self.loc.x then
                self.maze.stat.lock = 1
                self.maze.stat.alice_actions[1] = 1
            elseif self.maze.lamp and self.maze.lamp.loc.y == self.loc.y 
                and self.maze.lamp.loc.x == self.loc.x then
                self.maze.stat.lamp = 1
                self.maze.stat.alice_actions[2] = 1
            end
        end
        toggle_f(self)
    end

    self:update_states()
end

function SPLightKey:reset()
    parent.reset(self)
    if self.door then
        self.map:remove_item(self.door)
        self.door.loc.y = self.door_init_loc_y
        self.door.loc.x = self.door_init_loc_x
        self.map:add_item(self.door)
    end
    if self.lamp then
        self.lamp.attr._c = self.lamp_init_color
        self.lamp.attr.color = 'color' .. self.lamp.attr._c
    end
    if self.switch then
        self.switch.attr._c = self.switch_init_color
        self.switch.attr.color = 'color' .. self.switch.attr._c
    end
end

function SPLightKey:rand_loc_opposite_to(item)
    local sy, sx = self.map:get_empty_loc()
    local gy = item.loc.y
    local gx = item.loc.x
    while true do
        if self.wall == 'vertical' then
            if sx < self.door_x and gx > self.door_x then break end
            if sx > self.door_x and gx < self.door_x then break end
        else
            if sy < self.door_y and gy > self.door_y then break end
            if sy > self.door_y and gy < self.door_y then break end
        end
        sy, sx = self.map:get_empty_loc()
    end
    return sy, sx
end

function SPLightKey:build_wall()
    if torch.uniform() < 0.5 then
        self.wall = 'vertical'
        self.door_y = torch.random(self.map.height)
        self.door_x = torch.random(2, self.map.width - 1)
        for y = 1, self.map.height do
            if y ~= self.door_y then
                self:place_item({type='block'}, y, self.door_x)
            end
        end
    else
        self.wall = 'horizontal'
        self.door_x = torch.random(self.map.width)
        self.door_y = torch.random(2, self.map.height - 1)
        for x = 1, self.map.width do
            if x ~= self.door_x then
                self:place_item({type='block'}, self.door_y, x)
            end
        end
    end
    if self.push_block_door then
        self.door = self:place_item({_factory = PushableBlock},
            self.door_y, self.door_x)
    else
        self.door = self:place_item({type='door', open = 'open'},
            self.door_y, self.door_x)
    end

    -- need it for reset
    self.door_init_loc_y = self.door.loc.y
    self.door_init_loc_x = self.door.loc.x
end

function SPLightKey:update_states()
    if self.switch then
        if self.switch.attr._c == 2 then
            self.door.attr.open = 'open'
        else
            self.door.attr.open = nil
        end
    end

    if self.lamp then
        for _, item in pairs(self.items) do
            if item.type == 'agent' then
            elseif item.type == 'switch' and item.attr.controls == 'lamp' then
            elseif item.type == 'corner' then
            else
                if not item.attr._invisible_bak_done then
                    item.attr._invisible_bak = item.attr._invisible
                    item.attr._invisible_bak_done = true
                end
                if self.lamp.attr._c == 1 then
                    item.attr._invisible = true
                else
                    item.attr._invisible = item.attr._invisible_bak
                end
            end
        end
    end
end

function SPLightKey:update()
    parent.update(self)

    self:update_states()
    
    if not self.test_mode then
        if self.door and self.current_mind == 1 then
            if self.door.loc.y == self.agent.loc.y and self.door.loc.x == self.agent.loc.x then
                self.stat.door_cross = 1
                self.stat.alice_actions[3] = 1
            end
            if self.push_block_door then
                self.stat.push_dist = math.abs(self.door_init_loc_y - self.door.loc.y) 
                    + math.abs(self.door_init_loc_x - self.door.loc.x)
            end
        end
    end
end

function SPLightKey:get_terminal_reward()
    -- use this chance to record stat
    self.stat.alice_actions = torch.zeros(1 + 3 + 3 + 1)
    if self.stat.door_cross == 1 and self.stat.lamp == 1 and self.stat.lock == 1 then
        self.stat.alice_actions[1] = 1
    elseif self.stat.door_cross == 1 and self.stat.lamp == 1 then
        self.stat.alice_actions[2] = 1
    elseif self.stat.door_cross == 1 and self.stat.lock == 1 then
        self.stat.alice_actions[3] = 1
    elseif self.stat.lamp == 1 and self.stat.lock == 1 then
        self.stat.alice_actions[4] = 1
    elseif self.stat.door_cross == 1 then
        self.stat.alice_actions[5] = 1
    elseif self.stat.lock == 1 then
        self.stat.alice_actions[6] = 1
    elseif self.stat.lamp == 1 then
        self.stat.alice_actions[7] = 1
    else
        self.stat.alice_actions[8] = 1
    end

    if self.test_mode then
        if self:is_success() then
            return 1 - (self.t) * 0.1
        else
            return - (self.t) * 0.1
        end
    else
        return 0
    end
end
