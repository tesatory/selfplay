-- Copyright (c) 2015-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the BSD-style license found in the
-- LICENSE file in the root directory of this source tree. An additional grant
-- of patent rights can be found in the PATENTS file in the same directory.

paths.dofile('MazeBase.lua')
paths.dofile('OptsHelper.lua')
paths.dofile('GameFactory.lua')
paths.dofile('MazeBridge.lua')

-- self-play
paths.dofile('SPBase.lua')
paths.dofile('SPLightKey.lua')

local function init_factory(opts, vocab)
    g_opts = opts -- TODO use some of main configs
    local config = dofile(opts.games_config_path)
    config.visibility = opts.visibility
    local games = {}
    local helpers = {}

    games.SPLightKey = SPLightKey
    helpers.SPLightKey = OptsHelper

    local factory = GameFactory(config, vocab, games, helpers)
    factory.new_game = function(self)
        if opts.game == nil or opts.game == '' then
            return self:init_random_game()
        else
            return self:init_game(opts.game)
        end
    end
    return factory
end

local function init_vocab()
    local vocab = {}
    local ivocab = {}
    local nwords = 0

    local function vocab_add(word)
        if vocab[word] == nil then
            local ind = #ivocab + 1
            vocab[word] = ind
            ivocab[ind] = word
        end
    end

    -- general
    vocab_add('nil')
    vocab_add('agent')
    for i = 1, 1 do
        vocab_add('agent' .. i)
    end
    vocab_add('goal')
    vocab_add('goal1')
    vocab_add('obj1')
    for i = 1, 60 do
        vocab_add('time' .. i)
    end
    -- for i = 1, 3 do
    --     vocab_add('goal' .. i)
    --     vocab_add('task' .. i)
    --     for j = 1, 3 do
    --         vocab_add('task' .. i .. j)
    --     end
    -- end
    vocab_add('visited')
    vocab_add('block')
    vocab_add('water')
    vocab_add('info')
    vocab_add('task')
    vocab_add('same')
    vocab_add('door')
    vocab_add('open')
    vocab_add('closed')
    vocab_add('switch')
    for i = 1, 5 do
        vocab_add('color' .. i)
        vocab_add('increment' .. i)
    end
    vocab_add('corner')
    vocab_add('flag')
    vocab_add('lamp')
    vocab_add('pushable')
    vocab_add('swamp')
    vocab_add('test_mode')
    vocab_add('perform_task')

    -- for y = 1, 10 do
    --     for x = 1, 10 do
    --         vocab_add('ay' .. y .. 'x' .. x)
    --     end
    -- end

    -- for craft:
    -- for s = 1, 10 do
    --     vocab_add('item' .. s)
    -- end
    -- for s = 1, 3 do
    --     vocab_add('obj' .. s)
    --     vocab_add(tostring(s))
    -- end
    -- vocab_add('craftingtable')
    -- vocab_add('craftable')
    -- vocab_add('inventory')
    -- vocab_add('task')

    -- vocab_add('wood')
    -- vocab_add('grass')
    -- vocab_add('iron')
    -- vocab_add('plank')
    -- vocab_add('stick')
    -- vocab_add('cloth')
    -- vocab_add('rope')
    -- vocab_add('bridge')
    -- vocab_add('bed')
    -- vocab_add('axe')
    -- vocab_add('shear')
    -- vocab_add('toolshed')
    -- vocab_add('workbench')
    -- vocab_add('factory')

    return vocab, ivocab
end

local games = {}
games.init_vocab = init_vocab
games.init_factory = init_factory
return games
