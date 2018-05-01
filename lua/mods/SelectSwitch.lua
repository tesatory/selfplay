-- takes multiple input tables, and output one selected by given index

local SelectSwitch, parent = torch.class('nn.SelectSwitch', 'nn.Module')

function SelectSwitch:__init()
    parent.__init(self)
    self.gradInput = {}
    self.batch_mode = true
end


function SelectSwitch:updateOutput(input)
    -- 1st input the indexes
    s = input[1]:long()
    self.output = self.output or input[2].new()
    self.output:resizeAs(input[2])
    if self.batch_mode then
        for i = 1, input[2]:size(1) do
            self.output[i]:copy(input[s[i]+1][i])
        end
    else
        error()
    end
    return self.output
end


function SelectSwitch:updateGradInput(input, gradOutput)
    s = input[1]:long()
    self.gradInput[1] = nil
    for i = 2, #input do
        self.gradInput[i] = self.gradInput[i] or gradOutput.new()
        self.gradInput[i]:resizeAs(gradOutput)
        self.gradInput[i]:zero()
    end
    if self.batch_mode then
        for i = 1, input[2]:size(1) do
            self.gradInput[s[i]+1][i]:copy(gradOutput[i])
        end
    else
        error()
    end
    return self.gradInput
end
