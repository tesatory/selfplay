-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

paths.dofile('MazeItem.lua')
paths.dofile('MazeAgent.lua')
paths.dofile('MazeMap.lua')

local MazeBase = torch.class('MazeBase')

function MazeBase:__init(opts, vocab)
    self.map = MazeMap(opts)
    self.max_attributes = opts.max_attributes
    self.vocab = vocab    
    self.nwords = 0
    for _,_ in pairs(vocab) do
        self.nwords = self.nwords + 1 -- TODO
    end
    self.t = 0
    self.costs = {}
    self.push_action = opts.push_action
    self.crumb_action = opts.crumb_action
    self.flag_visited = opts.flag_visited
    self.enable_boundary = opts.enable_boundary
    self.enable_corners = opts.enable_corners

    -- This list contains EVERYTHING in the game.
    self.items = {}
    self.items_bytype = {}
    self.item_byname = {}

    if self.enable_boundary == 1 then
        self:add_boundary()
    end

    for i, j in pairs(opts.costs) do
        self.costs[i] = j
    end

    self.ngoals = opts.ngoals or 1
    self.nagents = opts.nagents or 1
    self.nblocks = opts.nblocks or 0
    self.nwater = opts.nwater or 0
    self.finished = false
    self.finish_by_goal = false
end


function MazeBase:add_default_items()
    for i = 1, self.nblocks do
        self:place_item_rand({type = 'block'})
    end
    for i = 1, self.nwater do
        local e = self:place_item_rand({type = 'water'})
    end
    if self.enable_corners == 1 then
        local H = self.map.height
        local W = self.map.width
        self:place_item({type = 'corner', _immaterial = true} ,1,1)
        self:place_item({type = 'corner', _immaterial = true} ,H,1)
        self:place_item({type = 'corner', _immaterial = true} ,1,W)
        self:place_item({type = 'corner', _immaterial = true} ,H,W)
    end

    if self.agent == nil then
        self.agents = {}
        for i = 1, self.nagents do
            self.agents[i] = self:place_item_rand({type = 'agent', name = 'agent' .. i})
        end
        self.agent = self.agents[1]
    end
end

function MazeBase:add_boundary()
    for x = 1, self.map.width do
        self:place_item({type = 'block'}, 1, x)
        self:place_item({type = 'block'}, self.map.height, x)
    end
    for y = 2, self.map.height-1 do
        self:place_item({type = 'block'}, y, 1)
        self:place_item({type = 'block'}, y, self.map.width)
    end
end

--rename add_prebuilt_item --> add_item
-- and add_item --> new_item  ?
function MazeBase:add_prebuilt_item(e)
    e.id = #self.items+1
    self.items[#self.items+1] = e
    if not self.items_bytype[e.type] then
        self.items_bytype[e.type] = {}
    end
    table.insert(self.items_bytype[e.type], e)
    if e.name then
        self.item_byname[e.name] = e
    end
    if e.loc then
        self.map:add_item(e)
    end
    return e
end

function MazeBase:add_item(attr)
    local e
    if attr._factory then
        e = attr._factory(attr,self)
    else
        if attr.type == 'agent' then
            e = MazeAgent(attr, self)
        else
            e = MazeItem(attr)
        end
    end
    self:add_prebuilt_item(e)
    return e
end

function MazeBase:place_item(attr, y, x)
    attr.loc = {y = y, x = x}
    return self:add_item(attr)
end

function MazeBase:place_item_rand(attr)
    local y, x = self.map:get_empty_loc()
    return self:place_item(attr, y, x)
end

function MazeBase:remove(item, l)
    for i = 1, #l do
        if l[i] == item then
            table.remove(l, i)
            break
        end
    end
end

function MazeBase:remove_item(item)
    if item.loc then
        self.map:remove_item(item)
    end
    if item.type then
        self:remove(item, self.items_bytype[item.type])
    end
    if item.name then
        self.item_byname[item.name] = nil
    end
    self:remove(item, self.items)
end

function MazeBase:remove_byloc(y, x, t)
    local l = self.map.items[y][x]
    local r = {}
    for i = 1, #l do
        if l[i].type == t then
            table.insert(r, l[i])
        end
    end
    for i = 1, #r do
        self:remove_item(r[i])
    end
end

function MazeBase:remove_bytype(type)
    local l = self.items_bytype[type]
    local r = {}
    for i = 1, #l do
        table.insert(r, l[i])
    end
    for i = 1, #r do
        self:remove_item(r[i])
    end
end

function MazeBase:remove_byname(name)
    self:remove_item(self.item_byname[name])
end

-- Agents call this function to perform action
function MazeBase:act(action)
    self.agent:act(action)
end

-- Update map state after each step
function MazeBase:update()
    self.t = self.t + 1
    for i = 1, #self.items do
        if self.items[i].updateable then
            self.items[i]:update(self)
        end
    end
    if self.finish_by_goal then
        local items = self.map.items[self.agent.loc.y][self.agent.loc.x]
        for i = 1, #items do
            if items[i].type == 'goal' then
                self.finished = true
            end
        end
    end
end

function MazeBase:to_sentence_item(e, sentence)
    local s = e:to_sentence(self.agent.loc.y, self.agent.loc.x)
    for i = 1, #s do
        sentence[i] = self.vocab[s[i]]
    end
end

function MazeBase:to_sentence_display()
    local count=0
    local sentence = ''
    for i = 1, #self.items do
        local e = self.items[i]
        if not e.attr._invisible then
            local s = e:to_sentence(self.agent.loc.y, self.agent.loc.x)
            for j = 1, #s do
                sentence =  sentence .. ' ' ..  s[j]
            end
        end
        sentence =  sentence .. '\n'
    end
    return sentence
end

-- Tensor representation that can be feed to a model
function MazeBase:to_sentence(sentence)
    local count=0
    local sentence = sentence or torch.Tensor(#self.items, self.max_attributes):fill(self.vocab['nil'])
    for i = 1, #self.items do
        if not self.items[i].attr._invisible then
            count= count + 1
            if count > sentence:size(1) then error('increase memsize!') end
            self:to_sentence_item(self.items[i], sentence[count])
        end
    end
    return sentence
end

function MazeBase:get_visible_state(data, use_lut, visibility, nwords)
    local lut_counter = 0
    for dy = -visibility, visibility do
        for dx = -visibility, visibility do
            local y, x
            if self.use_abs_loc then
                y = math.ceil(self.map.height / 2) + dy
                x = math.ceil(self.map.width / 2) + dx
            else
                y = self.agent.loc.y + dy
                x = self.agent.loc.x + dx
            end
            if self.map.items[y] and self.map.items[y][x] then
                for _, e in pairs(self.map.items[y][x]) do
                    if self.agent == e or (not e.attr._invisible) then
                        local s = e:to_sentence(0, 0, true)
                        for i = 1, #s do
                            if self.agent ~= e and s[i]:sub(1,4) == 'talk' then
                                -- ignore it
                            else
                                if self.vocab[s[i]] == nil then error('not found in dict:' .. s[i]) end
                                local data_y = dy + visibility + 1
                                local data_x = dx + visibility + 1
                                local p = (data_y-1)*(2*visibility+1) + data_x
                                if use_lut then
                                    lut_counter = lut_counter + 1
                                    if lut_counter > data:size(1) then
                                        error('increase encoder_lut_size!')
                                    end
                                    data[lut_counter] = (p-1)*nwords + self.vocab[s[i]]
                                else
                                    data[p][self.vocab[s[i]]] = 1
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    if self.items_bytype['info'] and #self.items_bytype['info'] > 0 then
        local info_counter = 0
        for k = 1, #self.items_bytype['info'] do
            local e = self.items_bytype['info'][k]
            if not e.attr._invisible then
                info_counter = info_counter + 1
                local s = e:to_sentence(0, 0, true)
                for i = 1, #s do
                    if self.vocab[s[i]] == nil then error('not found in dict:' .. s[i]) end
                    lut_counter = lut_counter + 1
                    if lut_counter > data:size(1) then
                        error('increase encoder_lut_size!')
                    end
                    local p = (2*visibility+1)^2 + info_counter
                    if use_lut then
                        data[lut_counter] = (p-1)*nwords + self.vocab[s[i]]
                    else
                        data[p][self.vocab[s[i]]] = 1
                    end
                end
            end
        end
    end
end

-- This reward signal is used for REINFORCE learning
function MazeBase:get_reward()
    local items = self.map.items[self.agent.loc.y][self.agent.loc.x]
    local reward = -self.costs.step
    for i = 1, #items do
        if items[i].type ~= 'agent' then
            if items[i].reward then
                reward = reward + items[i].reward
            elseif self.costs[items[i].type] then
                reward = reward - self.costs[items[i].type]
            end
        end
    end
    return reward
end

function MazeBase:is_active()
    return (not self.finished)
end

function MazeBase:is_success()
    if self:is_active() then
        return false
    else
        return true
    end
end

function MazeBase:is_active_bprop()
    return true
end

--------------------------------------------------------
-- some utilities for supervision:
--------------------------------------------------------
function MazeBase:flatten_cost_map()
    local H = self.map.height
    local W = self.map.width
    self.cmap = torch.ones(H,W):mul(self.costs.step)
    for i,j in pairs(self.items) do
        local btype  =  j.type
        if j.type == 'door' then
            if j.attr.open == 'closed' then
                btype = 'block'
            end
        end
        local c = self.costs[btype]
        if c then
            local y = j.loc.y
            local x = j.loc.x
            self.cmap[y][x] = math.max(c,self.cmap[y][x])
        end
    end
end


function MazeBase:search_move_and_update(dh,dw,states,ans,rew,acount)
    --finds the best path to dh,dw from current location with
    --Dijkstra then moves there.
    -- if distance to target > 800
    if not self.ds then
        local ds = paths.dofile('search.lua')
        self.ds = ds
    end
    if not self.cmap then
         self:flatten_cost_map()
    end
    local sh = self.agent.loc.y
    local sw =  self.agent.loc.x
    if sh==dh and sw==dw then return acount,true end
    local dist,prev = self.ds.dsearch(self.cmap,sh,sw,dh,dw)
    if dist> 800 then return acount,false end
    local path = self.ds.backtrack(sh,sw,dh,dw,prev)
    local pl = path:size(1) - 1
    for t = 1, pl do
        acount = acount + 1
        local dy = path[t+1][1]-path[t][1]
        local dx = path[t+1][2]-path[t][2]
        ans[acount]  = self:d2a(dy,dx)
        states[acount] = self:to_sentence()
        self:act(ans[acount])
        self:update()
        rew[acount] = self:get_reward()
    end
    return acount, true
end

function MazeBase:d2a(dy,dx)
    local lact
    if dy< 0 then
        lact = 'up'
    elseif dy> 0 then
        lact = 'down'
    elseif dx> 0 then
        lact = 'right'
    elseif dx< 0 then
        lact = 'left'
    end
    return self.agent.action_ids[lact]
end
