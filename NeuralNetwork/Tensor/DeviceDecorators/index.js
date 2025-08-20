import * as CPU from './CPU.js'
import * as WGPU from './WebGPU.js'
import * as WORKER from './Worker.js'


export const DEVICE_TYPES = {
    CPU: 'cpu',
    WGPU: 'wgpu',
    WOKRER: 'worker',
}

const id_var = 'device_Id'
const type_var = 'device_Type'

function decoratorProxy(device) {
    let module;
    switch (device[type_var]) {
        case DEVICE_TYPES.CPU:
            module = CPU
            break;
        case DEVICE_TYPES.WGPU:
            module = WGPU
            break;
        case DEVICE_TYPES.WOKRER:
            module = WORKER
            break;
        default:
            break;
    }

    const handler = {
        get(target, property, receiver) {
            if (property == id_var || property == type_var) return target[property]
            if (module[property]) return (...args) => module[property](target, ...args)
            else {
                const func = target[property]

                if (typeof func === 'function') return func.bind(target)
                else return func
            }
        },
        set() {}
    }

    return new Proxy(device, handler)
}

const CPU_DEVICE = {}
CPU_DEVICE[type_var] = DEVICE_TYPES.CPU
CPU_DEVICE[id_var] = 0

const CPU_PROXY = decoratorProxy(CPU_DEVICE)

export function requestCPUDevice() {
    return CPU_PROXY
}

export async function requestWGPUDevice(id = Math.random().toString(36).substring(2)) {
    if (!navigator.gpu) {
      console.error("WebGPU not supported on this browser.");
      return;
    }

    // Request WebGPU Adapter & Device
    const adapter = await navigator.gpu.requestAdapter();
    console.log(adapter.features.has("shader-f16"))
    const device = await adapter.requestDevice({
        // requiredFeatures: ["subgroups"],
        requiredFeatures: ["shader-f16"],
    });

    device[type_var] = DEVICE_TYPES.WGPU
    device[id_var] = id

    WGPU.registerDevice(device, id)

    return decoratorProxy(device);
}
