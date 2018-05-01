-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

local MazeBridge = torch.class('MazeBridge')

function MazeBridge:__init(factory, opts)
    self.opts = opts
    self.factory = factory
end

function MazeBridge:batch_init(size)
    local batch = {}
    for i = 1, size do
        batch[i] = self.factory:new_game()
    end
    assert(self.opts.nactions == batch[1].agent.nactions, 
        'set nactions=' .. batch[1].agent.nactions)
    assert(self.opts.nagents == batch[1].nagents)
    return batch
end

function MazeBridge:batch_input(batch)
    if self.opts.encoder_lut then
        return self:batch_input_lut(batch)
    end
    local indim = (2*self.opts.visibility+1)^2 + self.opts.max_info
    local input = torch.Tensor(#batch, self.opts.nagents, indim, self.opts.nwords)
    input:fill(0)
    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            set_current_agent(g, a)
            g:get_visible_state(input[i][a], false, self.opts.visibility)
        end
    end
    input = input:view(#batch * self.opts.nagents, -1)
    return input
end

function MazeBridge:batch_input_lut(batch)
    local input = torch.Tensor(#batch, self.opts.nagents, self.opts.encoder_lut_size)
    input:fill(self.opts.encoder_lut_nil)
    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            set_current_agent(g, a)
            g:get_visible_state(input[i][a], true, self.opts.visibility, self.opts.nwords)
        end
    end
    input = input:view(#batch * self.opts.nagents, -1)
    return input
end

function MazeBridge:batch_act(batch, action, active)
    action = action:view(-1)
    active = active:view(#batch, self.opts.nagents)
    action = action:view(#batch, self.opts.nagents)
    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            set_current_agent(g, a)
            if active[i][a] == 1 then
                g:act(action[i][a])
            end
        end
    end
end

function MazeBridge:batch_reward(batch, active)
    active = active:view(#batch, self.opts.nagents)
    local reward = torch.Tensor(#batch, self.opts.nagents):zero()
    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            set_current_agent(g, a)
            if active[i][a] == 1 then
                reward[i][a] = g:get_reward()
            end
        end
    end
    return reward:view(-1)
end

function MazeBridge:batch_terminal_reward(batch)
    local reward = torch.Tensor(#batch, self.opts.nagents):zero()
    for i, g in pairs(batch) do
        if g.get_terminal_reward then
            for a = 1, self.opts.nagents do
                set_current_agent(g, a)
                reward[i][a] = g:get_terminal_reward()
            end
        end
    end
    return reward:view(-1)
end

function MazeBridge:batch_update(batch, active)
    active = active:view(#batch, self.opts.nagents)
    for i, g in pairs(batch) do
        if active[i]:sum() > 0 then
            g:update()
        end
    end
end

function MazeBridge:batch_active(batch)
    local active = torch.Tensor(#batch, self.opts.nagents):zero()
    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            set_current_agent(g, a)
            if g:is_active() then
                active[i][a] = 1
            end
        end
    end
    return active:view(-1)
end

function MazeBridge:batch_success(batch)
    local success = torch.Tensor(#batch):fill(0)
    for i, g in pairs(batch) do
        if g:is_success() then
            success[i] = 1
        end
    end
    return success
end
