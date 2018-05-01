-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

require('image')
local colors_ok, colors = pcall(paths.dofile, 'ansicolors.lua')
local MazeMap = torch.class('MazeMap')

function MazeMap:__init(opts)
    self.height = opts.map_height or 10
    self.width = opts.map_width or 10
    self.img_path = './games/images/'

    -- Items by x,y location
    self.items = {}
    for y = 1, self.height do
        self.items[y] = {}
        for x = 1, self.width do
            self.items[y][x] = {}
        end
    end

    self.visibility_mask = torch.Tensor(self.height, self.width)
    self.visibility_mask:fill(1)
end

function MazeMap:add_item(item)
    table.insert(self.items[item.loc.y][item.loc.x], item)
end

function MazeMap:remove_item(item)
    local l = self.items[item.loc.y][item.loc.x]
    for i = 1, #l do
        if l[i].id == item.id then
            table.remove(l, i)
            break
        end
    end
end

function MazeMap:get_empty_loc(fat)
    local fat = fat or 0
    local x, y
    for i = 1, 100 do
        y = torch.random(1+fat, self.height-fat)
        x = torch.random(1+fat, self.width-fat)
        local empty = true
        for j, e in pairs(self.items[y][x]) do
            if not e.attr._immaterial then
                empty = false
            end
        end
        if empty then return y, x end
    end
    error('failed 100 times to find empty location')
end

function MazeMap:is_loc_reachable(y, x)
     if y < 1 or x < 1 then
        return false
    elseif y > self.height or x > self.width then
        return false
    end
    local l = self.items[y][x]
    local is_reachable = true
    for i = 1, #l do
        is_reachable = is_reachable and l[i]:is_reachable()
    end
    return is_reachable
end

function MazeMap:is_loc_visible(y, x)
    if self.visibility_mask[y][x] == 1 then
        return true
    else
        return false
    end
end

local function read_png(fname)
    local im = image.load(fname)
    if im:size(1) == 4 then
        im = im:sub(1,3)
    elseif im:size(1) == 1 then
        im = im:expand(3,im:size(2),im:size(3)):clone()
    elseif im:size(1) == 2 then
        im = im:sub(2,2)
    im = im:expand(3,im:size(2),im:size(3)):clone()
    end
    return im
end

function MazeMap:to_image()
    local K = 32
    local img = torch.Tensor(3, self.height * K, self.width * K):fill(1)
    local img_goals={}
    for s = 1, 9 do
        img_goals[s]= read_png(self.img_path .. 'goal' .. s .. '.png')
    end
    local img_block = read_png(self.img_path .. 'block.png')
    local img_water = read_png(self.img_path .. 'water.png')
    local img_pushableblock = read_png(self.img_path .. 'pushableblock.png')
    local img_bfire = read_png(self.img_path .. 'blue_fire.png')
    local img_rfire = read_png(self.img_path .. 'red_fire.png')
    local img_agent = read_png(self.img_path .. 'agent.png')

    local img_starenemy = {}
    img_starenemy[1] = read_png(self.img_path .. 'starenemy1.png')
    img_starenemy[2] = read_png(self.img_path .. 'starenemy2.png')
    img_starenemy[3] = read_png(self.img_path .. 'starenemy3.png')
    img_starenemy[4] = read_png(self.img_path .. 'starenemy4.png')
    img_starenemy[5] = read_png(self.img_path .. 'starenemy5.png')

    local item_images = {}
    item_images['item1'] = read_png(self.img_path .. 'wood.png')
    item_images['item2'] = read_png(self.img_path .. 'iron.png')
    item_images['item3'] = read_png(self.img_path .. 'stone.png')
    item_images['item4'] = read_png(self.img_path .. 'axe.png')
    item_images['item5'] = read_png(self.img_path .. 'hammer.png')
    item_images['item6'] = read_png(self.img_path .. 'sword.png')
    item_images['item7'] = read_png(self.img_path .. 'bridge.png')
    item_images['item4T'] = read_png(self.img_path .. 'anvil.png')
    item_images['item5T'] = read_png(self.img_path .. 'workbench.png')
    item_images['item6T'] = read_png(self.img_path .. 'furnace.png')
    item_images['item7T'] = read_png(self.img_path .. 'factory.png')

    for y = 1, self.height do
        for x = 1, self.width do
            local c = img:narrow(2,1+(y-1)*K,K):narrow(3,1+(x-1)*K,K)
            for i = 1, #self.items[y][x] do
                local item = self.items[y][x][i]
                if not item.attr._invisible then
                    if item.type == 'block' then
                        c:copy(img_block)
                    elseif item.type == 'pushableblock' then
                        c:copy(img_pushableblock)
            		elseif item.type == 'water' then
                        c:copy(img_water)
                    elseif item.type == 'goal' then
                        for a = 1, 9 do
                            if item.name == 'goal' .. a then
                                c:copy(img_goals[a])
                            end
                        end
                    elseif item.type == 'door' then
                         if item.attr._c == 1 then
                            c[1]:fill(0)
                        elseif item.attr._c == 2 then
                            c[2]:fill(0)
                        elseif item.attr._c == 3 then
                            c[3]:fill(0)
                        else
                            c:fill(0.5)
                        end
                         if item.attr.open=='open' then
                             c[1]:narrow(2,1,K/2):fill(0)
                             c[2]:narrow(2,1,K/2):fill(1)
                             c[3]:narrow(2,1,K/2):fill(0)
                         else
                             c[1]:narrow(2,1,K/2):fill(1)
                             c[2]:narrow(2,1,K/2):fill(0)
                             c[3]:narrow(2,1,K/2):fill(0)
                         end
                    elseif item.type == 'agent' then
                        c:copy(img_agent)
                    elseif item.type == 'StarEnemy' then
                        c[1]:fill(1)
                        c[2]:fill(0)
                        c[3]:fill(0)
                        for a = 1, 5 do
                            if item.name == 'enemy' .. a then
                                c:copy(img_starenemy[a])
                            end
                        end
                    elseif item.type == 'switch' then
                        if item.attr._c == 1 then
                            c[1]:fill(0)
                        elseif item.attr._c == 2 then
                            c[2]:fill(0)
                        elseif item.attr._c == 3 then
                            c[3]:fill(0)
                        else
                            c:fill(0.5)
                        end
                    elseif item.type == 'crumb' then
                        c[{ 1, {15,18}, {15,18} }]:fill(1)
                        c[{ 2, {15,18}, {15,18} }]:fill(0)
                        c[{ 3, {15,18}, {15,18} }]:fill(0)
                    elseif item.type == 'craftable' then
                        local name = item.attr.name
                        if item_images[name] then
                            c:copy(item_images[name])
                        else
                            c:fill(0.5)
                        end
                    elseif item.type == 'craftingtable' then
                        local name = item.attr.name
                        if item_images[name .. 'T'] then
                            c:copy(item_images[name .. 'T'])
                        else
                            c:fill(0.5)
                        end
                    end
                end
            end
            if self.ygoal and self.ygoal == y and self.xgoal == x then
                c[2]:add(.4)
            end
        end
    end
    if self.agent_shots_d then
        for s = 1, #self.agent_shots_d do
            local l = self.agent_shots_d[s]
            for t = 1, l:size(1) do
                local y = l[t][1]
                local x = l[t][2]
                local c = img:narrow(2,1+(y-1)*K,K):narrow(3,1+(x-1)*K,K)
                c:copy(img_rfire:sub(1,3,1,-1,1,-1))
            end
        end
    end
    if self.enemy_shots_d then
        for s = 1, #self.enemy_shots_d do
            local l = self.enemy_shots_d[s]
            for t = 1, l:size(1) do
                local y = l[t][1]
                local x = l[t][2]
                local c = img:narrow(2,1+(y-1)*K,K):narrow(3,1+(x-1)*K,K)
                c:copy(img_bfire:sub(1,3,1,-1,1,-1))
            end
        end
    end

    local mask = self.visibility_mask:view(1, self.height, 1, self.width, 1)
    mask = mask:expand(3, self.height, K, self.width, K):clone()
    mask = mask:view(3, self.height * K, self.width * K)
    img:cmul(mask)
    return img
end

function MazeMap:print_ascii()
    for y = 0, self.height + 1 do
        local line = '|'
        for x = 1, self.width do
            local s = '  '
            if y == 0 or y == self.height + 1 then
                s = '--'
            else
                for i = 1, #self.items[y][x] do
                    local item = self.items[y][x][i]
                    if item.attr._ascii then
                        s = item.attr._ascii:sub(1,2)
                    elseif item.type == 'block' then
                        s = '[]'
                    elseif item.type == 'goal' then
                        s = '$$'
                    elseif item.type == 'agent' then
                        s = '@@'
                    elseif item.type == 'marine' then
                        s = 'MA'
                    elseif item.type == 'medic' then
                        s = 'ME'
                    elseif item.type == 'door' then
                        if item.attr.open == 'open' then
                            s = 'DO'
                        else
                            s = 'DC'
                        end
                    elseif item.type == 'switch' then
                        if item.attr.controls then
                            s = item.attr.controls:sub(1,1) .. item.attr._c
                        else
                            s = 'S' .. item.attr._c
                        end
                    elseif item.type == 'corner' then
                        
                    else
                        s = '??'
                    end
                    if colors_ok and item.attr._invisible then
                        s = colors('%{dim}' .. s)
                    elseif item.attr._ascii_color then
                        s = colors('%{' .. table.concat(item.attr._ascii_color, ' ') .. '}' .. s)
                    elseif item.type == 'block' then
                        s = colors('%{red}' .. s)
                    elseif item.type == 'agent' then
                        s = colors('%{blue}' .. s)
                    elseif item.type == 'goal' then
                        if item.attr.visited == 'visited' then
                            s = colors('%{green}' .. s)
                        else
                            s = colors('%{yellow}' .. s)
                        end
                    end
                end
            end
            line = line .. s
        end
        line = line .. '|'
        print(line)
    end
end
