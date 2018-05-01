local GaussianNLL, parent = torch.class('nn.GaussianNLL', 'nn.Criterion')

function GaussianNLL:__init(sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
    self.gradInput = {}
end

function GaussianNLL:updateOutput(input, target)
    local mean = input[1]
    local log_std = input[2]
    local d = mean - target
    d:pow(2)
    d:cmul(torch.exp(- 2 * log_std))
    d:div(2)
    d:add(log_std)
    if self.sizeAverage then
        self.output = d:mean()
    else
        self.output = d:sum()
    end
    return self.output
end

function GaussianNLL:updateGradInput(input, target)
    local mean = input[1]
    local log_std = input[2]
    for i = 1, 2 do
        if self.gradInput[i] == nil then
            self.gradInput[i] = input[i].new()
            self.gradInput[i]:resizeAs(input[i])
        end
    end
    local gm = self.gradInput[1]
    gm:zero()
    gm:add(mean)
    gm:add(-1, target)
    gm:cmul(torch.exp(- 2 * log_std))

    local gs = self.gradInput[2]
    gs:zero()
    gs:add(mean)
    gs:add(-1, target)
    gs:pow(2)
    gs:cmul(torch.exp(- 2 * log_std))
    gs:mul(-1)
    gs:add(1)

    return self.gradInput
end
