'''
Run Length Coding 
Author: chunyangf@qq.com
'''
from ctypes import *
from tkinter import TRUE
from tkinter.messagebox import NO
import numpy as np
import os
import time

lib = cdll.LoadLibrary(
    os.path.dirname(os.path.abspath(__file__)) +
    '/resAc.so')  # class level loading lib
lib.encoding.restype = c_void_p
lib.encoding.argtypes = [
    POINTER(c_int32), c_int, c_int,
    POINTER(c_uint8),
    POINTER(c_uint32), c_bool
]
lib.decoding.restype = c_void_p
lib.decoding.argtypes = [
    POINTER(c_uint8), c_int, c_int, c_int,
    POINTER(c_int32)
]


def encode_res(data, detail=False):
    if np.ndim(data) == 1:
        channel = 1
    else:
        channel = data.shape[1]
    data = np.reshape(data.T, data.shape, order='C')
    data = np.ascontiguousarray(data).astype(np.int32)
    data_p = data.ctypes.data_as(POINTER(c_int32))
    code = (c_uint8 * (data.shape[0] * channel * 32))()
    code_len = c_uint32()
    lib.encoding(data_p, data.shape[0], channel, code, byref(code_len), detail)
    return bytes(np.array(code[0:code_len.value]).tolist())


def decode_res(code, pointNum, channel):
    code = np.ascontiguousarray(list(code)).reshape(-1).astype(np.uint8)
    code_p = code.ctypes.data_as(POINTER(c_uint8))
    data = (c_int32 * (pointNum * channel))()
    lib.decoding(code_p, code.shape[0], pointNum, channel, data)
    return np.array(data[0:pointNum * channel]).reshape(channel, pointNum).T

def encode_int128(value):
    return value.to_bytes(16, byteorder='little')

def decode_int128(encoded_value):
    return int.from_bytes(encoded_value, byteorder='little')

def encode_res_multichannel(data, detail=False):
    ave =  np.round(data.mean(0))
    data = data - ave
    # append ave at data end
    data = np.concatenate([data, ave[np.newaxis,:]], axis=0)
    channel = data.shape[1]
    chan3 = channel // 3
    bits = []
    bits.append(bytes([channel-chan3 * 3]))  # 1 channel num
    for c in range(chan3):
        encoded_chunk = encode_res(data[:, c*3:(c+1)*3])
        bits.append(encode_int128(len(encoded_chunk)))  
        bits.append(encoded_chunk)

    for c in range(chan3 * 3, channel):
        encoded_chunk = encode_res(data[:, c:c+1])
        bits.append(encode_int128(len(encoded_chunk)))  
        bits.append(encoded_chunk)

    concatenated_bits = b''.join(bits)
    return concatenated_bits

def decode_res_multichannel(code,pointNum):
    trunck_code = []
    channel = code[0]
    idx = 1
    while idx < len(code):
        length = decode_int128(code[idx:idx+16])
        idx += 16
        chunk = code[idx:idx+length]
        trunck_code.append(chunk)
        idx += length
    data = []
    chan3 = len(trunck_code) - channel
    for c in range(chan3):
        data.append(decode_res(trunck_code[c], pointNum+1, 3))
    for c in range(chan3, len(trunck_code)):
        data.append(decode_res(trunck_code[c], pointNum+1, 1))
    data = np.concatenate(data, axis=1)
    # ave
    ave = data[-1:,:]
    return data [:-1,:] + ave

if __name__ == "__main__":
    data = np.random.randint(-1000, 1000, (100000, 3))
    code = encode_res(data, True)
    data_d = decode_res(code, 100000, 3)
    print((data == data_d).all())

    data = np.random.randint(-1000, 1000, (100000, 56))
    code = encode_res_multichannel(data, True)
    data_d = decode_res_multichannel(code, 100000)
    print((data == data_d).all())
