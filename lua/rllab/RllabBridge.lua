require'image'
local py = paths.dofile('../bridge.lua')

local RllabBridge = torch.class('RllabBridge')

function RllabBridge:__init(opts)
    py.init()
    py.exec([=[
import sys
sys.path.append(path)
    ]=], {path = paths.dirname(paths.thisfile())})
    py.exec('from rllab_worker import *')
    py.eval('init(env_name, size, opts)', {
        env_name = opts.rllab_env, size = opts.batch_size, opts = opts})

    self.opts = opts
    assert(self.opts.nagents == 1)
    self.env_name = opts.rllab_env
    self.size = opts.batch_size
end

function RllabBridge:batch_init(size)
    assert(self.size == size)
    local batch = {}
    local obs = py.eval('reset()')
    for i = 1, size do
        batch[i] = {obs = obs[i], done = false, t = 0}
    end

    if not self.opts.rllab_cont_action and self.opts.nactions > 1 then
        assert(self.opts.nactions == self:get_nactions(),
            'set nactions=' .. self:get_nactions())
    end

    if self.opts.nminds > 1 then
        local mind = py.eval('current_mind()')
        for i, g in pairs(batch) do
            g.current_mind = mind[i]
        end
    end

    return batch
end

function RllabBridge:batch_input(batch)
    local input = torch.Tensor(self.size, self.opts.rllab_in_dim)
    for i, g in pairs(batch) do
        assert(self.opts.rllab_in_dim == g.obs:size(1), "set input dim to " .. g.obs:size(1))
        input[i]:copy(g.obs)
    end
    input = input:view(#batch, -1)

    if self.opts.rllab_normalize then
        self.obs_max = self.obs_max or torch.zeros(1, self.opts.rllab_in_dim):fill(0.01)
        self.obs_min = self.obs_min or torch.zeros(1, self.opts.rllab_in_dim):fill(-0.01)
        self.obs_max:cmax(torch.max(input, 1), self.obs_max)
        self.obs_min:cmin(torch.min(input, 1), self.obs_min)
        local max = self.obs_max:expandAs(input)
        local min = self.obs_min:expandAs(input)
        input = torch.cdiv(input - min, max - min) * 2 - 1
    end

    return input
end

function RllabBridge:batch_act(batch, action, active)
    if type(action) ~= 'table' then
        action = {action}
    end
    local rllab_action = torch.zeros(action[1]:size(1), self.opts.naction_heads)
    for k = 1, self.opts.naction_heads do
        local name = self.opts.action_names[k]
        -- when nminds>2, use the last action for switching between Alice and Bob
        if self.opts.rllab_cont_action and (self.opts.nminds == 1 or k < self.opts.naction_heads) then
            local d = torch.linspace(-self.opts.rllab_cont_limit, self.opts.rllab_cont_limit, self.opts.nactions_byname[name])
            for i = 1, action[k]:size(1) do
                rllab_action[i][k] = d[action[k][i][1]]
            end
        else
            if self.opts.nactions_byname[name] == 1 then
                action[k] = torch.clamp(action[k], -self.opts.rllab_cont_limit, self.opts.rllab_cont_limit)
            end
            for i = 1, action[k]:size(1) do
                rllab_action[i][k] = action[k][i][1] - 1
            end
        end
    end

    local obs, reward, done, info = unpack(py.eval('step(action, active, steps)',
        {action = rllab_action, active = active, steps = self.opts.rllab_steps}))
    for i, g in pairs(batch) do
        if active[i] == 1 then
            g.obs = obs[i]
            g.reward = reward[i]
            g.done = done[i]
            if self.opts.nminds > 1 then
                g.current_mind = info[i].current_mind
            end
            g.t = g.t + 1
        end
    end
end

function RllabBridge:batch_reward(batch, active)
    local reward = torch.Tensor(#batch):zero()
    for i, g in pairs(batch) do
        if active[i] == 1 then
            reward[i] = g.reward
        end
    end
    reward:mul(self.opts.rllab_reward_coeff)
    return reward
end

function RllabBridge:batch_terminal_reward(batch)
    local reward = torch.Tensor(#batch):zero()
    local r = py.eval('reward_terminal()')
    for i = 1, #batch do
        reward[i] = r[i]
    end
    reward:mul(self.opts.rllab_reward_coeff)

    if self.opts.nminds > 1 then
        for i, g in pairs(batch) do
            g.reward_terminal_mind = {}
            g.get_terminal_reward_mind = function(self, m)
                return self.reward_terminal_mind[m]
            end
        end
        for m = 1, self.opts.nminds do
            local mind = torch.Tensor(self.size):fill(m)
            local r = py.eval('reward_terminal_mind(mind)', {mind = mind})
            for i, g in pairs(batch) do
                g.reward_terminal_mind[m] = r[i] * self.opts.rllab_reward_coeff
            end
        end
    end

    local stat = py.eval('get_stat()')
    for i, g in pairs(batch) do
        g.stat = stat[i]
        g.type = g.stat.type
        g.stat.type = nil
        if g.stat.test_pos or g.stat.switch_pos then
            if __threadid == nil or __threadid == 1 then
                local x = torch.zeros(1, 6)
                if g.stat.test_pos then
                    x:narrow(2,1,2):copy(g.stat.test_pos)
                end
                if g.stat.switch_pos then
                    x:narrow(2,3,2):copy(g.stat.switch_pos)
                    x:narrow(2,5,2):copy(g.stat.final_pos)
                end
                g.stat.position = {op = 'join', data = x}
            end
            g.stat.switch_pos = nil
            g.stat.test_pos = nil
            g.stat.final_pos = nil
        end
        g.success = g.stat.success
        g.stat.success = nil
    end

    return reward
end

function RllabBridge:batch_update(batch, active)
end

function RllabBridge:batch_active(batch)
    local active = torch.Tensor(#batch):zero()
    for i, g in pairs(batch) do
        if not g.done then
            active[i] = 1
        end
    end
    return active
end

function RllabBridge:batch_success(batch)
    local success = torch.Tensor(#batch):fill(0)
    for i, g in pairs(batch) do
        if g.success then
            success[i] = 1
        end
    end
    return success
end

function RllabBridge:render()
    py.exec('render()')
end

-- to make video
-- ffmpeg -framerate 15 -i swim_%d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p out.mp4
function RllabBridge:save_image(g, path)
    local img = py.eval('render(True)')
    img = img:permute(3, 1, 2):div(256)
    image.save(path .. '_' .. g.t .. '.png', img)
end

function RllabBridge:get_nactions()
    assert(self.opts.rllab_cont_action == false)
    return py.eval('num_actions()')
end
