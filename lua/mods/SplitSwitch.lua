-- splits input to N way, using given index

local SplitSwitch, parent = torch.class('nn.SplitSwitch', 'nn.Module')

function SplitSwitch:__init(N)
    parent.__init(self)
    self.output = {}
    self.gradInput = {}
    self.batch_mode = true
    self.N = N
end


function SplitSwitch:updateOutput(input)
    local x, s = unpack(input)
    for i = 1, self.N do
        self.output[i] = self.output[i] or x.new()
        self.output[i]:resizeAs(x)
        self.output[i]:zero()
    end
    s = s:long()
    if self.batch_mode then
        for i = 1, x:size(1) do
            self.output[s[i]][i]:copy(x[i])
        end
    else
        error()
    end
    return self.output
end


function SplitSwitch:updateGradInput(input, gradOutput)
    local x, s = unpack(input)
    s = s:long()
    self.gradInput[1] = self.gradInput[1] or gradOutput[1].new()
    self.gradInput[1]:resizeAs(gradOutput[1])

    if self.batch_mode then
        for i = 1, x:size(1) do
            self.gradInput[1][i]:copy(gradOutput[s[i]][i])
        end
    else
        error()
    end
    return self.gradInput
end
