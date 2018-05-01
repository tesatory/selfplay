local SCBattleEnv, parent = torch.class('SCBattleEnv', 'SCEnv')

function SCBattleEnv:__init(opts, torchcraft, vocab)
    parent.__init(self, opts, torchcraft, vocab)

    self.enemy_ind2uid = {}
    for uid, ut in pairs(self.state.units_enemy) do
        self.enemy_ind2uid[#self.enemy_ind2uid+1] = uid
    end
    self:update_state()

    assert(self.opts.nactions >= 4 + #self.enemy_ind2uid)
end

function SCBattleEnv:update_state()
    parent.update_state(self)
    self.hp_total_myself = 0
    for i, uid in pairs(self.agent_ind2uid) do
        if self.agents[i] then
            self.hp_total_myself = self.hp_total_myself + self.agents[i].hp
        end
    end
    self.enemies = {}
    self.hp_total_enemy = 0
    if self.enemy_ind2uid then
        for i, uid in pairs(self.enemy_ind2uid) do
            self.enemies[i] = self.state.units_enemy[uid]
            if self.enemies[i] then
                self.hp_total_enemy = self.hp_total_enemy + self.enemies[i].hp
            end
        end
    end
end

function SCBattleEnv:get_terminal_reward(agent_ind)
    if self.hp_total_myself > self.hp_total_enemy then
        return 1
    else
        return 0
    end
end

function SCBattleEnv:act(agent_ind, action)
    local agent_uid = self.agent_ind2uid[agent_ind]
    local agent = self.agents[agent_ind]
    local x = agent.position[1]
    local y = agent.position[2]

    local offset =32

    -- move around + attack
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
        -- attack a target
        local target_uid = self.enemy_ind2uid[action - 4]
        if target_uid ~= nil then
            table.insert(self.actions_buffer, self.tc.command(self.tc.command_unit_protected,
                agent_uid, self.tc.cmd.Attack_Unit, target_uid))
        end
    end
end
