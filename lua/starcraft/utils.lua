local tc_utils = require'torchcraft.utils'

local utils = {}
utils.get_closest = function(position, unitsTable)
    local min_d = 1E30
    local closest_uid = nil
    for uid, ut in pairs(unitsTable) do
        local tmp_d = tc_utils.distance(position, ut['position'])
        if tmp_d < min_d then
            min_d = tmp_d
            closest_uid = uid
        end
    end
    return closest_uid
end

return utils