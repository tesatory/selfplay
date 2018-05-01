--
local Switch, parent = torch.class('nn.Switch', 'nn.Module')

function Switch:__init(ignore_grad)
    parent.__init(self)
    self.gradInput = {}
    self.ignore_grad = ignore_grad or false
end


function Switch:updateOutput(input)
    local x, s = unpack(input)
    local sz = x:size()
    sz[2] = 1
    s = s:view(-1, 1, 1):expand(sz)
    s = s:long()
    self.output = x:gather(2, s)
    self.output = self.output:view(-1, sz[3])
    return self.output
end


function Switch:updateGradInput(input, gradOutput)
    local x, s = unpack(input)
    local sz = x:size()
    sz[2] = 1
    s = s:view(-1, 1, 1):expand(sz)
    s = s:long()
    self.gradInput[1] = self.gradInput[1] or gradOutput.new()
    self.gradInput[1]:resizeAs(x)
    self.gradInput[1]:zero()
    if self.ignore_grad == false then
        gradOutput = gradOutput:view(-1, 1, sz[3])
        self.gradInput[1]:scatter(2, s, gradOutput)
    end
    return self.gradInput
end
