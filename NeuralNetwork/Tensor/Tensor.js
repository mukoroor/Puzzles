/**
 * @typedef {Object} Tensor
 * @property {TypedArray|GPUBuffer} data
 * @property {Int32Array} size
 * @property {TypedArray} type
 * @property {GPUBuffer} [gpu_size]
 * @property {Object} device
 */

import { DEVICE_TYPES, requestCPUDevice } from './DeviceDecorators/index.js';

const HOST_TYPES = {
    GPU_DEVICE_HOST: { baseName: 'GPU_DEVICE' , allowedTypes:  {
        Float32Array,
        Int32Array,
        Uint32Array,
    } },
}
const TypedArray = Object.getPrototypeOf(Int8Array);

const tensorHandler = {
    get(target, property, receiver) {
        if (property == 'deviceType') {
          return target.device.deviceType
        }
        return Reflect.get(target, property, receiver);
    },

    set(target, property, value, reciever) {
        return
    }
}

export async function toString(t) {
    return `Tensor {
        size: ${new Array(t.size)},
        data: ${await t.device.toString(data, t.type)}
        type: ${t.type.name}
        device: ${t.deviceType}
        }`
    }
    
export async function sendToDevice(t, device = requestCPUDevice()) {
    let data = t.data
    if (t.deviceType != DEVICE_TYPES.CPU) data = await t.device.sendToMainThread(t.data, t.size, t.type);

    return from(data, t.size, t.type, device);
}
    
function tensorProxy(tensorData) { return new Proxy(tensorData, tensorHandler) }
    
function validate(data, size) {
    return data.length <= size.reduce((a, c) => a * c);
}

export function from(data, size, type = Float32Array, device = requestCPUDevice()) {
    if (!data) return;

    size = new Uint32Array(size || [data.length, 1]);

    if (!validate(data, size)) return;

    if (!(data instanceof TypedArray)) {
        data = (type).from(data)
    }

    data = device.initialize(data, size, 'tensor_data', 'ab')
    const gpu_size = device.type === DEVICE_TYPES.WGPU ? device.initialize(size, [2], 'tensor_size'): null

    return tensorProxy({ data, size, type, gpu_size, device })
}

export function empty(size, type = Float32Array, device = requestCPUDevice()) {
    size = new Uint32Array(size);

    const data = device.allocate([...size, type.BYTES_PER_ELEMENT], 'tensor_data')
    const gpu_size = device.type === DEVICE_TYPES.WGPU ? device.initialize(size, [2], 'tensor_size'): null

    return tensorProxy({ data, size, type, gpu_size, device })
}

export async function clone(t) {
    const [data, size, gpu_size] = await Promise.all([
        t.device.clone(t.data),
        t.device.clone(t.size),
        (t.gpu_size ? t.device.clone(t.size) : null),
    ])

    return tensorProxy({
        ...t,
        data,
        size,
        gpu_size
    })
}

async function convert(t, TypedArray) {
    const [data, clonedTensor] = await Promise.all([
        t.device.convert(t.data, TypedArray),
        clone(t),
    ])

    return tensorProxy({
        ...clonedTensor,
        data,
    })
}

export async function transform(t, mapFn) {
    const [data, clonedTensor] = await Promise.all([
        t.device.transform(t.data, mapFn),
        clone(t),
    ])

    return tensorProxy({
        ...clonedTensor,
        data,
    })
}

export async function round(t) {
    return await transform(t, Math.round)
}

export async function trunc(t) {
    return await transform(t, Math.trunc)
}

export async function floor(t) {
    return await transform(t, Math.floor)
}

export async function ceil(t) {
    return await transform(t, Math.ceil)
}

// Int
export async function toInt64(t) {
    return await convert(t, BigInt64Array);
}

export async function toInt32(t) {
    return await convert(t, Int32Array);
}

export async function toInt16(t) {
    return await convert(t, Int16Array);
}

export async function toInt8(t) {
    return await convert(t, Int8Array);
}

export const toInt = toInt32;

// Uint
export async function toUint64(t) {
    return await convert(t, BigUint64Array);
}

export async function toUint32(t) {
    return await convert(t, Uint32Array);
}

export async function toUint16(t) {
    return await convert(t, Uint16Array);
}

export async function toUint8(t) {
    return await convert(t, Uint8Array);
}

export const toUint = toUint32;

// Float
export async function toFloat64(t) {
    return await convert(t, Float64Array);
}

export async function toFloat32(t) {
    return await convert(t, Float32Array);
}

export const toFloat = toFloat32;
