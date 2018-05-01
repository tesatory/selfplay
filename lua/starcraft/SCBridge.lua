require 'sys'
require 'os'
paths.dofile('SCEnv.lua')
paths.dofile('SCBattleEnv.lua')
paths.dofile('SCEconEnv.lua')
paths.dofile('SCSPEconEnv.lua')

local SCBridge = torch.class('SCBridge')
local DEBUG = 0

function SCBridge:__init(opts)
    self.opts = opts
    self.tc_conns = {}
    self:init_vocab()
end

function SCBridge:connect(tc, port)
    tc:init(self.opts.sc_host, port)
    local update = tc:connect(port)
    if DEBUG > 1 then
        print('Received init: ', update)
    end
    assert(tc.state.replay == false)

    -- first message to BWAPI's side is setting up variables
    local gui = 0 
    local skip = 10000 
    if self.opts.batch_size == 1 and self.opts.nworker == 1 then 
        gui = 1 
        skip = 1
    end
    local setup = {
        tc.command(tc.set_speed, 0),
        tc.command(tc.set_combine_frames, 1 + self.opts.sc_skip_frames),
        tc.command(tc.set_gui, gui),
        tc.command(tc.set_frameskip, skip),
        tc.command(tc.set_cmd_optim, 1),
    }
    tc:send({table.concat(setup, ':')})
end

function SCBridge:close(tc)
    tc:close()
    sys.sleep(0.5)
end

function SCBridge:init_vocab()
    local vocab = {}
    local ivocab = {}

    local function vocab_add(word)
        if vocab[word] == nil then
            local ind = #ivocab + 1
            vocab[word] = ind
            ivocab[ind] = word
        end
    end

    -- general
    vocab_add('nil')
    vocab_add('team_myself')
    vocab_add('team_neutral')
    vocab_add('team_enemy')
    vocab_add('agent')
    vocab_add('enemy')
    vocab_add('Resource_Mineral_Field')
    vocab_add('Terran_SCV')
    vocab_add('Terran_Command_Center')
    vocab_add('Terran_Barracks')
    vocab_add('Terran_Marine')
    vocab_add('Terran_Supply_Depot')
    vocab_add('ore')
    self.opts.vocab_num_limit = 40
    for i = 0, self.opts.vocab_num_limit do
        vocab_add('num' .. i)
    end
    -- flags
    vocab_add('idle')
    vocab_add('gathering_minerals')
    vocab_add('constructing')
    vocab_add('training')
    vocab_add('being_gathered')
    vocab_add('being_constructed')
    -- self-play
    vocab_add('test_mode')
    vocab_add('self_play')


    self.vocab = vocab
    self.ivocab = ivocab
    self.nwords = #self.ivocab
end

function SCBridge:batch_init(size)
    self.size = size
    local thread_port = self.opts.sc_port
    if __threadid then
        -- assume all threads are the same size
        thread_port = thread_port + size * (__threadid - 1)
    end
    if #self.tc_conns < size then
        for i = #self.tc_conns + 1, size do
            local port = thread_port + i - 1
            -- os.execute('cd ~/StarCraft && TORCHCRAFT_PORT=' .. port .. ' ../TorchCraft/BWEnv/build/bwenv &')
            -- os.execute('sleep 5')
            -- os.execute('sleep ' .. (thread_port + i - self.opts.sc_port) / 20)
            self.tc_conns[i] = require 'torchcraft'.new()
            self.tc_conns[i].DEBUG = DEBUG
            -- Enables a battle ended flag, for when one side loses all units
            self.tc_conns[i].micro_battles = true
            self:connect(self.tc_conns[i], port)
            self.tc_conns[i].restart_needed = true -- TODO
        end
    end
    for i = 1, size do
        if self.tc_conns[i].state.game_ended then
            self:close(self.tc_conns[i])
            self:connect(self.tc_conns[i], thread_port + i - 1)
            self.tc_conns[i].restart_needed = true -- TODO
        end
    end

    -- restart starcraft
    for i = 1, size do
        if self.tc_conns[i].restart_needed then
            if DEBUG > 1 then print('restarting begin') end
            local tc = self.tc_conns[i]
            local actions
            while actions == nil do
                if DEBUG > 1 then print('looking for Zerg_Infested_Terran') end
                local update = tc:receive()
                tc:send({''})
                for uid, ut in pairs(tc.state.units_myself) do
                    if ut.type == tc.unittypes.Zerg_Infested_Terran then
                        actions = {tc.command(tc.command_unit, uid, tc.cmd.Move, 0, 22, 33)}
                        if DEBUG > 1 then print(actions) end
                        break
                    end
                end
            end
            while tc.state.battle_just_ended == false do
                local update = tc:receive()
                if tc.DEBUG > 1 then
                    print('restarting: ',
                        tc.state.battle_just_ended,
                        tc.state.waiting_for_restart,
                        tc.state.game_ended)
                end
                tc:send({table.concat(actions, ':')})
            end
        end
        self.tc_conns[i].restart_needed = true
    end

    local batch = {}
    for i = 1, size do
        if self.opts.sc_game == 'econ' then
            batch[i] = SCEconEnv(self.opts, self.tc_conns[i], self.vocab)
        elseif self.opts.sc_game == 'sp_econ' then
            batch[i] = SCSPEconEnv(self.opts, self.tc_conns[i], self.vocab)
        else
            error('wrong game')
        end
    end

    return batch
end

function SCBridge:batch_input(batch)
    local indim = (2*self.opts.visibility+1)^2 + self.opts.max_info
    local input
    if self.opts.encoder_lut then
        input = torch.Tensor(#batch, self.opts.nagents, self.opts.encoder_lut_size)
        input:fill(self.opts.encoder_lut_nil)
    else
        input = torch.Tensor(#batch, self.opts.nagents, indim, self.opts.nwords)
        input:fill(0)
    end

    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            g:observe(a, input[i][a], self.opts.visibility)
        end
    end
    input = input:view(#batch * self.opts.nagents, -1)
    return input
end

function SCBridge:batch_act(batch, action, active)
    active = active:view(#batch, self.opts.nagents)
    action = action:view(#batch, self.opts.nagents)
    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            if active[i][a] == 1 then
                g:act(a, action[i][a])
            end
        end
    end
end

function SCBridge:batch_update(batch, active)
    active = active:view(#batch, self.opts.nagents)
    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            if active[i][a] == 1 then
                g:update()
                break
            end
        end
    end
end

function SCBridge:batch_active(batch)
    local active = torch.Tensor(#batch, self.opts.nagents):zero()
    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            if g:is_active(a) then
                active[i][a] = 1
            end
        end
    end
    return active:view(-1)
end

function SCBridge:batch_reward(batch, active)
    active = active:view(#batch, self.opts.nagents)
    local reward = torch.Tensor(#batch, self.opts.nagents):zero()
    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            if active[i][a] == 1 then
                reward[i][a] = g:get_reward(a)
            end
        end
    end
    return reward:view(-1)
end

function SCBridge:batch_terminal_reward(batch)
    local reward = torch.Tensor(#batch, self.opts.nagents):zero()
    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            reward[i][a] = g:get_terminal_reward(a)
        end
    end
    return reward:view(-1)
end

function SCBridge:batch_success(batch)
    local success = torch.Tensor(#batch, self.opts.nagents):fill(0)
    for i, g in pairs(batch) do
        for a = 1, self.opts.nagents do
            success[i][a] = 0
        end
    end
    return success:view(-1)
end
