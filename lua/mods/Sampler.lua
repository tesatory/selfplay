--
local Sampler, parent = torch.class('nn.Sampler', 'nn.Module')

function Sampler:__init(keep_long)
    parent.__init(self)
    self.keep_long = keep_long or false
end

local function sample_multinomial(p)
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

-- input must be log(prob)
function Sampler:updateOutput(input)
    assert(input:dim() == 2)
    self.output = sample_multinomial(torch.exp(input))
    if self.output_fixed then
        for i = 1, self.output_fixed:size(1) do
            if self.output_fixed[i][1] > 0 then
                self.output[i][1] = self.output_fixed[i][1]
            end
        end
    end
    if self.keep_long then
        return self.output
    else
        return self.output:float()
    end
end

function Sampler:updateGradInput(input, gradOutput)
    -- ignore gradOutput
    self.gradInput = self.gradInput or input.new()
    self.gradInput:resizeAs(input)
    self.gradInput:zero()
    if self.cost then
        self.gradInput:scatter(2, self.output, self.cost)
    end
    if self.entropy_reg then
        local g = torch.exp(input):cmul(torch.add(input, 1))
        g:cmul(self.entropy_reg:view(-1, 1):expandAs(g))
        self.gradInput:add(g)
    end
    if self.output_fixed then
        for i = 1, self.output_fixed:size(1) do
            if self.output_fixed[i][1] > 0 then
                self.gradInput[i]:zero()
            end
        end
    end
    return self.gradInput
end
