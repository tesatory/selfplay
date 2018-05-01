require('os')
require('sys')
local struct = require('struct')
local stringx = require('pl.stringx')

local bridge = {}

local zmq = require('lzmq')


local function byte2tensor(bytes)
  local dim, ind = struct.unpack('l', bytes:sub(1, 8))
  local size = {}
  for i=1, dim do
    size[#size+1] = struct.unpack('l', bytes:sub(i*8+1, i*8+9))
  end
  bytes = bytes:sub(dim*8+9)
  local f = torch.MemoryFile()
  f:binary()
  f:writeString(bytes)
  f:seek(1)
  local tensor = torch.DoubleTensor(unpack(size))
  tensor:storage():read(f)
  return tensor
end

local function deserialize(arg)
  local KEY = "TENSOR_BYTES"
  local OUT = {}
  local TENSORLST = {}
  while true do
    local s, e = arg:find(KEY)
    if s == nil then break end
    OUT[#OUT+1] = arg:sub(1, s-1)
    local len = struct.unpack('l', arg:sub(e+1, e+8))
    TENSORLST[#TENSORLST+1] = byte2tensor(arg:sub(e+9, e+9+len-1))
    arg = arg:sub(e+9+len)
  end
  local STRING_TO_LOAD = ""
  for i=1, #OUT do
    STRING_TO_LOAD = STRING_TO_LOAD .. OUT[i] .. "TENSORLST[" .. i .. "]"
  end
  STRING_TO_LOAD = STRING_TO_LOAD .. arg
  local code = assert(loadstring("return " .. STRING_TO_LOAD))
  setfenv(code, {TENSORLST=TENSORLST})
  return code()
end

local function serialize(arg)
  if type(arg) == "string" then
    return '"' .. arg .. '"'
  elseif type(arg) == "number" then
    return arg
  elseif type(arg) == "boolean" then
    return (arg and "True" or "False")
  elseif type(arg) == "table" then
    if arg[1] ~= nil and arg[0] == nil and arg[-1] == nil then
      local lst = '['
      for i=1, #arg do
        lst = lst .. serialize(arg[i]) .. ','
      end
      return lst .. ']'
    else
      local tbl = '{'
      for k, v in pairs(arg) do
        tbl = tbl .. serialize(k) .. ':' .. serialize(v) .. ','
      end
      return tbl .. '}'
    end
  elseif type(arg) == 'userdata' and string.find(arg:type(), "Tensor") then
    -- Always serialized to a double array for now
    local f = torch.MemoryFile()
    f:binary()
    f:writeLong(arg:dim())
    for i = 1, arg:dim() do
      f:writeLong(arg:size(i))
    end
    arg:double():contiguous():storage():write(f)
    data = f:storage():string()
    return 'TENSOR_BYTES' .. struct.pack('l', #data) .. data
  else
    error("Cannot serialize this type")
  end
end

function bridge.get_free_port()
  -- TCP ports linger 2 minutes after they're closed, kind of annoying
  while true do
    local port = torch.random(20000, 65000)
    local context = zmq.context()
    local requester, err = context:socket{zmq.REQ}
    local success, err = requester:bind("tcp://*:" .. port)
    if success then
      requester:close()
      return port
    end
    print(port .. ' is busy')
  end
end

function bridge.init()
  local port = bridge.get_free_port()
  os.execute(string.format("python3 worker.py %d &", port))

  local context = zmq.context()
  local requester, err = context:socket{zmq.REQ,
    connect = "tcp://localhost:" .. port
  }
  bridge.conn = requester
  bridge.conn:send("a")
  assert(bridge.conn:recv() == "connected")
end


function bridge.is_connected()
  bridge.conn:send("a")
  if bridge.conn:poll(30) and bridge.conn:recv() == "connected" then
    return true
  end
  return false
end

function bridge.eval(code, args)
  -- Evaluate python code and returns serialized result
  -- Can serialize: lists, dicts, int, float, bool, string
  --      with arbitrary nesting
  if args then
    for k, v in pairs(args) do
      bridge.conn:send("x" .. k .. "=" .. serialize(v))
      bridge.conn:recv()
    end
  end
  bridge.conn:send("e"..code)

  return deserialize(bridge.conn:recv())
end

function bridge.exec(code, args)
  -- Evaluate python code and returns serialized result
  -- Can serialize: lists, dicts, int, float, bool, string
  --      with arbitrary nesting
  if args then
    for k, v in pairs(args) do
      bridge.conn:send("x" .. k .. "=" .. serialize(v))
      bridge.conn:recv()
    end
  end
  bridge.conn:send("x"..code)
  bridge.conn:recv()
end

return bridge
