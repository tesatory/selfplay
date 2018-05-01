--
local GaussianSampler, parent = torch.class('nn.GaussianSampler', 'nn.Module')

function GaussianSampler:__init(keep_long)
    parent.__init(self)
    self.keep_long = keep_long or false
end


function GaussianSampler:updateOutput(input)
    self.output = self.output or input.new()
    self.output:resize(input:size(1), 1)
    for i = 1, input:size(1) do
        self.output[i][1] = torch.normal() * math.exp(input[i][2]) + input[i][1]
    end
    return self.output
end

function GaussianSampler:updateGradInput(input, gradOutput)
    -- ignore gradOutput
    self.gradInput = self.gradInput or input.new()
    self.gradInput:resizeAs(input)
    self.gradInput:zero()
    if self.cost then
        for i = 1, input:size(1) do
            local mu = input[i][1]
            local std = math.exp(input[i][2])
            local x = self.output[i][1]
            self.gradInput[i][1] = self.cost[i][1] * (x - mu)/(std^2)
            self.gradInput[i][2] = self.cost[i][1] * ((x - mu)^2/(std^2) - 1)
        end
    end

    -- if self.entropy_reg then
    --     local g = torch.exp(input):cmul(torch.add(input, 1))
    --     g:cmul(self.entropy_reg:view(-1, 1):expandAs(g))
    --     self.gradInput:add(g)
    -- end

    return self.gradInput
end
