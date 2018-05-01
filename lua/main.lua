local function split(inputstr, sep)
    local t={}
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[#t+1] = str
    end
    return t
end

require 'sys'
paths.dofile('Trainer.lua')

local cmd = torch.CmdLine()
-- model parameters
cmd:option('--hidsz', 50, 'the size of the internal state vector')
cmd:option('--nonlin', 'tanh', 'non-linearity type: tanh | relu | none')
cmd:option('--init_std', 0.2, 'STD of initial weights')
cmd:option('--init_hid', 0.1, 'initial value of internal state')
cmd:option('--encoder_lut', false, 'use LookupTable in encoder instead of Linear')
cmd:option('--encoder_lut_size', 50, 'max items in encoder LookupTable')
cmd:option('--recurrent', false, 'recurrent model')
cmd:option('--nlayers', 2, 'number of hidden layers')
-- game parameters
cmd:option('--nagents', 1, 'the number of agents')
cmd:option('--coop', false, 'fully coop game')
cmd:option('--nactions', '6', 'the number of agent actions (1 for contineous). Use N:M:K for multiple actions')
cmd:option('--max_steps', 20, 'force to end the game after this many steps')
cmd:option('--games_config_path', 'games/config/find.lua', 'configuration file for games')
cmd:option('--game', '', 'can specify a single game')
cmd:option('--visibility', 1, 'vision range of agents. Does not apply to MemNN')
cmd:option('--max_info', 0, 'max number of info items allowed')
-- training parameters
cmd:option('--mode', 'pg', 'actor-critic or policy-grad: ac | pg')
cmd:option('--ac_freq', 1, 'how often to boostrap in AC')
cmd:option('--optim', 'rmsprop', 'optimization method: rmsprop | sgd')
cmd:option('--lrate', 3e-3, 'learning rate')
cmd:option('--alpha', 0.1, 'coefficient of baseline term in the cost function')
cmd:option('--epochs', 100, 'the number of training epochs')
cmd:option('--nbatches', 10, 'the number of mini-batches in one epoch')
cmd:option('--batch_size', 16, 'size of mini-batch (the number of parallel games) in each thread')
cmd:option('--nworker', 16, 'the number of threads used for training')
cmd:option('--entropy_reg', 0, 'entropy regularization for action probs')
cmd:option('--entropy_reg_decay', 0, 'decay rate')
cmd:option('--constant_baseline', -1, 'if in (0,1) use a scalar baseline instead of nn model')
-- for optim
cmd:option('--momentum', 0, 'momentum for SGD')
cmd:option('--wdecay', 0, 'weight decay for SGD')
cmd:option('--rmsprop_alpha', 0.97, 'parameter of RMSProp')
cmd:option('--rmsprop_eps', 1e-6, 'parameter of RMSProp')
cmd:option('--adam_beta1', 0.9, 'parameter of Adam')
cmd:option('--adam_beta2', 0.999, 'parameter of Adam')
cmd:option('--adam_eps', 1e-8, 'parameter of Adam')
cmd:option('--max_grad_norm', 0, 'clip gradient norm')
-- for RLLab
cmd:option('--rllab', false, 'use rllab')
cmd:option('--rllab_env', 'MountainCar')
cmd:option('--rllab_in_dim', 2)
cmd:option('--rllab_reward_coeff', 1)
cmd:option('--rllab_steps', 1, 'rllab steps between actions (i.e. skip frame)')
cmd:option('--rllab_cont_action', false, 'convert to contineous action')
cmd:option('--rllab_cont_limit', 1, 'action value is between [-x, x]')
cmd:option('--rllab_normalize', false, 'normalize input')
cmd:option('--rllab_normalize_rllab', false, 'normalize input by RLLab')
cmd:option('--rllab_save_image', '', 'path to save rendered image')
-- starcraft
cmd:option('--sc', false, 'use starcraft')
cmd:option('--sc_game', 'sp_econ', 'game type')
cmd:option('--sc_resolution', 4, '')
cmd:option('--sc_host', '172.17.0.2', 'starcraft host server')
cmd:option('--sc_port', 11111, 'use ports starting at')
cmd:option('--sc_skip_frames', 23)
cmd:option('--sc_count_alpha', 0, 'do count-base exploration in starcraft')

--other
cmd:option('--save', '', 'file name to save the model')
cmd:option('--save_all', false, 'save every epochs')
cmd:option('--load', '', 'file name to load the model')
cmd:option('--load_opts', false, 'load arg options too')
cmd:option('--show', false)
cmd:option('--hand', false)
cmd:option('--debug', false)
cmd:option('--plot', false)
cmd:option('--plot_env', 'main')
cmd:option('--rand_seed', 0)
cmd:option('--curriculum_sta', 0, 'start making harder after this many epochs')
cmd:option('--curriculum_end', 0, 'when to make the game hardest')
-- multi mind mode (i.e. multiple policies)
cmd:option('--nminds', 1)
cmd:option('--mind_reward_separate', false, 'reset cumulative reward when mind changes')
cmd:option('--minds_share_enc', false)
cmd:option('--mind_target', false, '')
-- self-play
cmd:option('--sp_mode', 'reverse', 'self-play mode: reverse | repeat | compete')
cmd:option('--sp_test_rate', 0.1, 'percentage of target task episodes')
cmd:option('--sp_test_rate_bysteps', false, 'test rate in number of steps instead of episodes')
cmd:option('--sp_test_steps_max', 0, 'terminate training after N test steps')
cmd:option('--sp_test_entr_zero', false, 'no entropy regularization for test task')
cmd:option('--sp_reward_bob_step', false, 'penalize Bob at every step instead of final reward')
-- for only mazebase tasks
cmd:option('--sp_rand_alice', 0, '1=randomize actions of alice')
-- for only for rllab tasks
cmd:option('--sp_state_thres', 0.1, 'threshold of success in contineous state')
cmd:option('--sp_reward_coeff', 0.01, 'multiply rewards in self-play mode')
cmd:option('--sp_loc_only', false, 'return only location (only swimmer gather)')
cmd:option('--sp_test_max_steps', 0, 'max steps for test tasks (swim only)')

local opts = cmd:parse(arg or {})

if string.find(opts.nactions, ':') then
    opts.nactions_byname = {}
    opts.naction_heads = 0
    opts.action_names = {}
    for i, n in pairs(string.split(opts.nactions, ':')) do
        opts.nactions_byname['action' .. i] = tonumber(n)
        opts.naction_heads = opts.naction_heads + 1
        opts.action_names[i] = 'action' .. i
    end
    opts.nactions = 0
else
    opts.nactions = tonumber(opts.nactions)
    opts.naction_heads = 1
    opts.nactions_byname = {action = opts.nactions}
    opts.action_names = {'action'}
end

if opts.load_opts then
    local opts_orig = opts
    local f = torch.load(opts.load)
    print(f.opts.nactions)
    opts = f.opts
    opts.save = ''
    opts.epochs = 0
    opts.load = opts_orig.load
    opts.plot = opts_orig.plot
    opts.plot_env = opts_orig.plot_env
end

print(opts)

if opts.plot then
    local visdom = require'visdom'
    g_plot = visdom{server = 'http://localhost', port = 8097, env = opts.plot_env}
    g_plot.ipv6 = false
end
if opts.rand_seed > 0 then torch.manualSeed(opts.rand_seed) end

if opts.debug then
    opts.nworker = 1
	opts.nbatches = 1
    dbg = require'debugger'
end

if opts.rllab then
    if opts.nminds > 1 then
        assert(opts.rllab_steps == 1, 'to correct switch action')
    end
end

trainer = Trainer(opts)
if opts.load ~= '' then
	trainer:load(opts.load)
end

if opts.debug then
    local dbg_train = function()
		trainer:train(opts.epochs)
    end
	local dbg_test = function()
		trainer:test()
    end
	dbg.call(dbg_test)
    dbg.call(dbg_train)
else
	trainer:train(opts.epochs)
end

if opts.save ~= '' then
	trainer:save(opts.save)
end
