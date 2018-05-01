local SCEconEnv, parent = torch.class('SCEconEnv', 'SCEnv')
local tc_utils = require'torchcraft.utils'
local utils = paths.dofile('utils.lua')

function SCEconEnv:__init(opts, torchcraft, vocab)
    parent.__init(self, opts, torchcraft, vocab)

    self.mineral_ids = {}
    self.units_mineral = self.tc:filter_type(self.tc.state.units_neutral,
        {self.tc.unittypes.Resource_Mineral_Field,
        self.tc.unittypes.Resource_Mineral_Field_Type_2,
        self.tc.unittypes.Resource_Mineral_Field_Type_3})
    for uid, ut in pairs(self.units_mineral) do
        self.mineral_ids[#self.mineral_ids+1] = uid
    end
    self:update_state()

    assert(self.opts.nactions == 7)
end

function SCEconEnv:is_unit_ready(ut)
    if ut.type == self.tc.unittypes.Terran_Barracks and ut.flags.being_constructed then
        return false
    elseif ut.type == self.tc.unittypes.Terran_Supply_Depot and ut.flags.being_constructed then
        return false
    elseif ut.type == self.tc.unittypes.Terran_SCV and ut.flags.completed == false then
        return false
    elseif ut.type == self.tc.unittypes.Terran_Marine and ut.flags.completed == false then
        return false
    end
    return true
end

function SCEconEnv:get_agents()
    local units = self.tc:filter_type(self.tc.state.units_myself, {
        self.tc.unittypes.Terran_SCV,
        self.tc.unittypes.Terran_Command_Center,
        self.tc.unittypes.Terran_Barracks
        })

    self.units_not_ready = 0
    for uid, ut in pairs(units) do
        if not self:is_unit_ready(ut) then
            units[uid] = nil
            self.units_not_ready = self.units_not_ready + 1
        end
    end

    return units
end

function SCEconEnv:observe_unit(obs, team, ut, loc_ind)
    local flags = {'idle'}
    if team == 'team_enemy' then
        -- ignore enemy
        return
    elseif ut.type == self.tc.unittypes.Zerg_Infested_Terran then
        -- ignore it (used for reset)
        return
    elseif ut.type == self.tc.unittypes.Resource_Vespene_Geyser then
        -- TODO
        return
    elseif ut.type == self.tc.unittypes.Terran_SCV then
        table.insert(flags, 'gathering_minerals')
        table.insert(flags, 'constructing')
        table.insert(flags, 'training')
    elseif ut.type == self.tc.unittypes.Resource_Mineral_Field or
        ut.type == self.tc.unittypes.Resource_Mineral_Field2 or
        ut.type == self.tc.unittypes.Resource_Mineral_Field3 then
        table.insert(flags, 'being_gathered')
    elseif ut.type == self.tc.unittypes.Terran_Barracks then
        table.insert(flags, 'being_constructed')
    elseif ut.type == self.tc.unittypes.Terran_Supply_Depot then
        table.insert(flags, 'being_constructed')
    end
    parent.observe_unit(self, obs, team, ut, loc_ind)
    for _, f in pairs(flags) do
        if ut.flags[f] then
            self:observe_word(obs, f, loc_ind)
        end
    end
end

function SCEconEnv:get_state_snapshot()
    local state = {}
    state.ore = math.floor(self.state.resources_myself.ore / 25)
    state.ore = math.min(state.ore, self.opts.vocab_num_limit)
    state.scvs = self.stat.unit_counts.Terran_SCV or 0
    state.barracks = self.stat.unit_counts.Terran_Barracks or 0
    state.marines = self.stat.unit_counts.Terran_Marine or 0
    state.depots = self.stat.unit_counts.Terran_Supply_Depot or 0
    return state
end

function SCEconEnv:observe_state(state, obs)
    self:observe_word(obs, 'num' .. state.ore, obs.info_loc)
    obs.info_loc = obs.info_loc + 1

    self:observe_word(obs, 'num' .. state.scvs, obs.info_loc)
    obs.info_loc = obs.info_loc + 1

    self:observe_word(obs, 'num' .. state.barracks, obs.info_loc)
    obs.info_loc = obs.info_loc + 1

    self:observe_word(obs, 'num' .. state.marines, obs.info_loc)
    obs.info_loc = obs.info_loc + 1

    self:observe_word(obs, 'num' .. state.depots, obs.info_loc)
    obs.info_loc = obs.info_loc + 1
end

function SCEconEnv:observe_info(obs)
    parent.observe_info(self, obs)

    local state = self:get_state_snapshot()
    self:observe_state(state, obs)
end

function SCEconEnv:update_state()
    parent.update_state(self)

    self.units_mineral = self.tc:filter_type(self.tc.state.units_neutral,
        {self.tc.unittypes.Resource_Mineral_Field,
        self.tc.unittypes.Resource_Mineral_Field_Type_2,
        self.tc.unittypes.Resource_Mineral_Field_Type_3})

    self.minerals = {}
    if self.mineral_ids then
        for i, uid in pairs(self.mineral_ids) do
            self.minerals[i] = self.units_mineral[uid]
        end
    end

    self.stat.unit_counts = {}
    for uid, ut in pairs(self.tc.state.units_myself) do
        if self:is_unit_ready(ut) then
            local type = self.tc.const.unittypes[ut.type]
            self.stat.unit_counts[type] = (self.stat.unit_counts[type] or 0) + 1
        end
    end

    self.nagent_in_build = 0
end

function SCEconEnv:get_terminal_reward(agent_ind)
    return self.stat.unit_counts.Terran_Marine or 0
end

function SCEconEnv:get_reward(agent_ind)
    if self.opts.sc_count_alpha > 0 then
        local s = self:get_state_snapshot()
        local r = self.opts.sc_count_alpha
        local n = self.state_counts[s.ore+1][s.scvs+1][s.barracks+1][s.depots+1][s.marines+1]
        if agent_ind == 1 then
	    n = n + 1
            self.state_counts[s.ore+1][s.scvs+1][s.barracks+1][s.depots+1][s.marines+1] = n
        end
        r = r / math.sqrt(n)
        return r
    else
        return 0
    end
end


function SCEconEnv:can_build_agent()
    local N = #self.agent_ind2uid
    N = N + self.nagent_in_build
    N = N + self.units_not_ready
    N = N + 1 -- TODO: slack of 1 unit
    return N < self.opts.nagents
end

function SCEconEnv:act(agent_ind, action)
    local agent_uid = self.agent_ind2uid[agent_ind]
    local agent = self.agents[agent_ind]

    if agent.type == self.tc.unittypes.Terran_SCV then
        local x = agent.position[1]
        local y = agent.position[2]
        local offset = 32

        -- move around + mine
        if action == 1 then
            table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit_protected,
                agent_uid, self.tc.cmd.Move, -1, x + offset, y, -1))
            self.stat.action_move = (self.stat.action_move or 0) + 1
        elseif action == 2 then
            table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit_protected,
                agent_uid, self.tc.cmd.Move, -1, x - offset, y, -1))
            self.stat.action_move = (self.stat.action_move or 0) + 1
        elseif action == 3 then
            table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit_protected,
                agent_uid, self.tc.cmd.Move, -1, x, y + offset, -1))
            self.stat.action_move = (self.stat.action_move or 0) + 1
        elseif action == 4 then
            table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit_protected,
                agent_uid, self.tc.cmd.Move, -1, x, y -offset, -1))
            self.stat.action_move = (self.stat.action_move or 0) + 1
        elseif action == 5 then
            -- mine the closest mineral
            if tc_utils.is_in(agent.order, self.tc.command2order[self.tc.unitcommandtypes.Gather]) then
                -- don't interrupt if already mining
            else
                local target_uid = utils.get_closest(agent.position, self.units_mineral)
                if target_uid ~= nil then
                    local d = tc_utils.distance(agent.position, self.units_mineral[target_uid].position)
                    if d <= 12 then
                        table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit_protected,
                            agent_uid, self.tc.cmd.Right_Click_Unit, target_uid))
                    end
                end
            end
            self.stat.action_mine = (self.stat.action_mine or 0) + 1
        elseif action == 6 then
            -- build a barrack
            if tc_utils.is_in(agent.order, self.tc.command2order[self.tc.unitcommandtypes.Build]) then
                -- don't interrupt if already building
            elseif self:can_build_agent() and self.state.resources_myself.ore >= 150 then
                table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit,
                    agent_uid, self.tc.cmd.Build, -1, agent.position[1], agent.position[2],
                    self.tc.unittypes.Terran_Barracks))
                self.stat.action_build = (self.stat.action_build or 0) + 1
                self.nagent_in_build = self.nagent_in_build + 1
            end
        elseif action == 7 then
            -- build a supply depot
            if tc_utils.is_in(agent.order, self.tc.command2order[self.tc.unitcommandtypes.Build]) then
                -- don't interrupt if already building
            elseif self.state.resources_myself.ore >= 100 then
                table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit,
                    agent_uid, self.tc.cmd.Build, -1, agent.position[1], agent.position[2],
                    self.tc.unittypes.Terran_Supply_Depot))
                self.stat.action_build = (self.stat.action_build or 0) + 1
            end
        else
            error('wrong action')
        end

    elseif agent.type == self.tc.unittypes.Terran_Command_Center then
        if action == 1 then
            -- train SCV
            if agent.idle and self:can_build_agent() and self.state.resources_myself.ore >= 50 then
                table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit, -- TODO: why not protected?
                    agent_uid, self.tc.cmd.Train, 0, 0, 0, self.tc.unittypes.Terran_SCV))
                self.stat.action_train = (self.stat.action_train or 0) + 1
                self.nagent_in_build = self.nagent_in_build + 1
            end
        end
    elseif agent.type == self.tc.unittypes.Terran_Barracks then
        if action == 1 then
            if agent.idle and self.state.resources_myself.ore >= 50 then
                table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit,
                    agent_uid, self.tc.cmd.Train, 0, 0, 0, self.tc.unittypes.Terran_Marine))
                self.stat.action_train = (self.stat.action_train or 0) + 1
            end
        end
    else
        error('unknown unit type: ' .. self.tc.const.unittypes[agent.type])
    end
end
