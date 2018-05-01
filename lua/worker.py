import numpy as np
import collections
import numbers
import zmq
import sys
import six
import traceback
import struct


def parseTensors(bytes):
    key = b"TENSOR_BYTES"
    out = []
    tensorlst = []
    while True:
        ind = bytes.find(key)
        if ind == -1: break
        out.append(bytes[:ind])
        length = struct.unpack('q', bytes[ind+len(key) : ind+len(key)+8])[0]
        tensorlst.append(byte2np(bytes[ind+len(key)+8:ind+len(key)+8+length]))
        bytes = bytes[ind+len(key)+8+length:]
    out.append(bytes)
    return "".join(s.decode() +
                   ("TENSORLIST[{}]".format(i) if i+1 != len(out) else "")
                   for i, s in enumerate(out)), tensorlst

def byte2np(bytes):
    dim = struct.unpack('q', bytes[:8])[0]
    shape = [0 for i in range(dim)]
    for d in range(dim):
        shape[d] = struct.unpack('q', bytes[8+d*8:8+d*8+8])[0]
    arr = np.frombuffer(bytes[8+d*8+16:-1], dtype='double').reshape(shape)
    return arr

def serialize(x):
    key = b"TENSOR_BYTES"
    if x is None:
        return b'nil'
    elif isinstance(x, six.string_types):
        return b'"' + x.encode() + b'"'
    elif type(x) == bool:
        return b'true' if x else b'false'
    elif isinstance(x, numbers.Number):
        return str(x).encode()
    elif isinstance(x, np.ndarray):
        arr = struct.pack('q', x.ndim)
        for d in x.shape:
            arr += struct.pack('q', d)
        arr = arr + struct.pack('q', x.size) + x.astype('double').tobytes()
        return key + struct.pack('q', len(arr)) + arr
    elif isinstance(x, collections.Mapping):
        return b'{' + b",".join([
            b''.join([b'[', serialize(k), b']=', serialize(v)])
            for k, v in x.items()]) + b'}'
    elif isinstance(x, collections.Iterable):
        return b'{' + b",".join(map(serialize, x)) + b'}'
    else:
        print("Cannot serialize variable of type {0}".format(type(x)))
        raise Exception("Cannot serialize variable of type {0}".format(type(x)))


def listener(port):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    print("binding to tcp://*:{}".format(port))
    socket.bind("tcp://*:{}".format(port))
    print("Server is listening...")
    while True:
        data = socket.recv()
        if data == "":
            break
        elif data[0] == ord('a'):
            res = b"connected"
        else:
            try:
                f = data[0]
                arg = data[1:]
                arg, TENSORLIST = parseTensors(arg)
                if f == ord('e'):
                    res = serialize(eval(arg))
                elif f == ord('x'):
                    exec(arg)
                    res = b'nil'
            except:
                res = b'nil'
                traceback.print_exc()
        socket.send(res)

if __name__ == "__main__":
    assert len(sys.argv) == 2
    listener(sys.argv[1])
