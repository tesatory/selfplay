function get_agent(g, a)
    return g.agents[a]
end

function set_current_agent(g, a)
    g.agent = get_agent(g, a)
end

function merge_stat(stat, s)
    for k, v in pairs(s) do
        if type(v) == 'number' then
            stat[k] = (stat[k] or 0) + v
        elseif type(v) == 'table' then
            if v.op == 'join' then
                if stat[k] then
                    local sz = stat[k].data:size()
                    sz[1] = sz[1] + v.data:size(1)
                    stat[k].data:resize(sz)
                    stat[k].data:narrow(1, sz[1]-v.data:size(1)+1, v.data:size(1)):copy(v.data)
                else
                    stat[k] = {data = v.data:clone(), op = v.op}
                end
            else
                if stat[k] == nil then stat[k] = {} end
                merge_stat(stat[k], v)
            end
        else
            -- it must be tensor
            if stat[k] then
                stat[k]:add(v)
            else
                stat[k] = v:clone()
            end
        end
    end
end

function sample_multinomial(p)
    -- for some reason multinomial fails sometimes
    local s, sample = pcall(
        function()
            return torch.multinomial(p, 1)
        end)
    if s == false then
        sample = torch.multinomial(torch.ones(p:size()),1)
    end
    return sample
end

function tensor_to_words(input, show_prob)
    for i = 1, input:size(1) do
        local line = i .. ':'
        for j = 1, input:size(2) do
            line = line .. '\t'  .. g_ivocab[input[i][j]]
        end
        if show_prob then
            for h = 1, g_opts.nhop do
                line = line .. '\t' .. string.format('%.2f', g_modules[h]['prob'].output[1][i])
            end
        end
        print(line)
    end
end


function format_stat(stat)
    local a = {}
    for n in pairs(stat) do table.insert(a, n) end
    table.sort(a)
    local str = ''
    for i,n in ipairs(a) do
        if string.find(n,'count_') then
            str = str .. n .. ': ' .. string.format("%2.4g",stat[n]) .. ' '
        end
    end
    str = str .. '\n'
    for i,n in ipairs(a) do
        if string.find(n,'reward_') then
            str = str .. n .. ': ' ..  string.format("%2.4g",stat[n]) .. ' '
        end
    end
    str = str .. '\n'
    for i,n in ipairs(a) do
        if string.find(n,'success_') then
            str = str .. n .. ': ' ..  string.format("%2.4g",stat[n]) .. ' '
        end
    end
    str = str .. '\n'
    -- str = str .. 'bl_cost: ' .. string.format("%2.4g",stat['bl_cost']) .. ' '
    str = str .. 'reward: ' .. string.format("%2.4g",stat['reward']) .. ' '
    str = str .. 'success: ' .. string.format("%2.4g",stat['success']) .. ' '
    str = str .. 'epoch: ' .. stat['epoch']
    return str
end
function print_tensor(a)
    local str = ''
    for s = 1, a:size(1) do str = str .. string.format("%2.4g",a[s]) .. ' '  end
    return str
end
function format_helpers(gname)
    local str = ''
    if not gname then
        for i,j in pairs(g_factory.helpers) do
            str = str .. i .. ' :: '
            str = str .. 'mapW: ' .. print_tensor(j.mapW) .. ' ||| '
            str = str .. 'mapH: ' .. print_tensor(j.mapH) .. ' ||| '
            str = str .. 'wpct: ' .. print_tensor(j.waterpct) .. ' ||| '
            str = str .. 'bpct: ' .. print_tensor(j.blockspct) .. ' ||| '
            str = str .. '\n'
        end
    else
        local j = g_factory.helpers[gname]
        str = str .. gname .. ' :: '
        str = str .. 'mapW: ' .. print_tensor(j.mapW) .. ' ||| '
        str = str .. 'mapH: ' .. print_tensor(j.mapH) .. ' ||| '
        str = str .. 'wpct: ' .. print_tensor(j.waterpct) .. ' ||| '
        str = str .. 'bpct: ' .. print_tensor(j.blockspct) .. ' ||| '
        str = str .. '\n'
    end
    return str
end

function plot_reward()
    local x = torch.zeros(#g_log)
    for i = 1, #g_log do
        x[i] = g_log[i].reward
    end
    gnuplot.plot(x)
end

function plot_stat(log, name, title, x_name)
    if type(name) ~= 'table' then name = {name} end
    title = title or name[1]
    local X = torch.linspace(1, #log, #log)
    local Y = torch.zeros(#log, #name)
    for i = 1, #log do
        for j = 1, #name do
            if log[i][name[j]] ~= nil then
                Y[i][j] = log[i][name[j]]
            end
        end
        if x_name then
            if log[i][x_name] then
                X[i] = log[i][x_name]
            else
                X[i] = 0
            end
        end
    end

    local xlabel = x_name or 'epoch'
    g_plot:line({Y=Y, X=X, win=title, options =
     {xlabel=xlabel, ylabel=title, legend=name}}
    )
end

function plot_switch_pos(s, opts)
    local x = torch.zeros(s:size(1) + 2, 2)
    if opts.rllab_env == 'SPMountainCar' then
        x[1][1] = -1
        x[1][2] = -1
        x[-1][1] = 1
        x[-1][2] = 1
    else
        x[1][1] = -4
        x[1][2] = -4
        x[-1][1] = 4
        x[-1][2] = 4
    end
    x:narrow(1,2,s:size(1)):copy(s:narrow(2,1,2))
    g_plot:scatter{X=x, win='test_pos', options={title='test_pos', markersize='3'}}
    x:narrow(1,2,s:size(1)):copy(s:narrow(2,3,2))
    g_plot:scatter{X=x, win='switch_pos', options={title='switch_pos', markersize='3'}}
    x:narrow(1,2,s:size(1)):copy(s:narrow(2,5,2))
    g_plot:scatter{X=x, win='final_pos', options={title='final_pos', markersize='3'}}
end

function play_pos()
    for i = 1, #trainer.log do
        print(i)
        plot_switch_pos(trainer.log[i].position.data, trainer.opts)
        os.execute('sleep 0.2')
    end
end
