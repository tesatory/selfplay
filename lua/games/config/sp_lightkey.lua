local opts = {}

opts.multigames = {}

local mapH = torch.Tensor{10,10,5,10,1}
local mapW = torch.Tensor{10,10,5,10,1}
local blockspct = torch.Tensor{0,.0,0,.2,.01}
local waterpct = torch.Tensor{0,.0,0,.2,.01}

-------------------
--some shared StaticOpts
local sso = {}
-------------- costs:
sso.costs = {}
sso.costs.goal = 0
sso.costs.step = 0.0
---------------------
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_boundary = 0
sso.enable_corners = 1
sso.with_wall = true
sso.with_switch = true
sso.test_rate = g_opts.sp_test_rate
sso.with_lamp = true
sso.return_loc_only = false
sso.lamp_off_prob = 0.3
sso.test_mind_separate = false -- use 3rd agent for test tasks
sso.test_rand_init = false  -- randomize test init states
sso.push_block_door = false -- replace door with a push-block

sso.nswamp = 0 
sso.swamp_stuck_prob = 0.9

-------------------------------------------------------
-- MultiGoals:
local SPLightKeyRangeOpts = {}
SPLightKeyRangeOpts.mapH = mapH:clone()
SPLightKeyRangeOpts.mapW = mapW:clone()
SPLightKeyRangeOpts.blockspct = blockspct:clone()
SPLightKeyRangeOpts.waterpct = waterpct:clone()

local SPLightKeyStaticOpts = {}
for i,j in pairs(sso) do SPLightKeyStaticOpts[i] = j end

local SPLightKeyOpts ={}
SPLightKeyOpts.RangeOpts = SPLightKeyRangeOpts
SPLightKeyOpts.StaticOpts = SPLightKeyStaticOpts

opts.multigames.SPLightKey = SPLightKeyOpts

return opts
