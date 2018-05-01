local GaussianCrossEntropy, parent = torch.class('nn.GaussianCrossEntropy', 'nn.Criterion')

function GaussianCrossEntropy:__init(sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
    self.gradInput = {}
end

function GaussianCrossEntropy:updateOutput(input, target)
    local mean = input[1]
    local log_std = input[2]
    local t_mean = target[1]
    local t_std = target[2]
    local d = mean - t_mean
    d:pow(2)
    d:add(torch.pow(t_std, 2))
    d:cmul(torch.exp(log_std:clone():mul(-2)))
    d:div(2)
    d:add(log_std)
    if self.sizeAverage then
        self.output = d:mean()
    else
        self.output = d:sum()
    end
    return self.output
end

function GaussianCrossEntropy:updateGradInput(input, target)
    local mean = input[1]
    local log_std = input[2]
    local t_mean = target[1]
    local t_std = target[2]
    for i = 1, 2 do
        if self.gradInput[i] == nil then
            self.gradInput[i] = input[i].new()
            self.gradInput[i]:resizeAs(input[i])
        end
    end
    local gm = self.gradInput[1]
    gm:zero()
    gm:add(mean)
    gm:add(-1, t_mean)
    gm:cmul(torch.exp(log_std:clone():mul(-2)))

    local gs = self.gradInput[2]
    gs:zero()
    gs:add(mean)
    gs:add(-1, t_mean)
    gs:pow(2)
    gs:add(torch.pow(t_std, 2))
    gs:cmul(torch.exp(log_std:clone():mul(-2)))
    gs:mul(-1)
    gs:add(1)

    return self.gradInput
end
