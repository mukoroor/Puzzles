
export function allocate() {}

export function write() {}

export function schedule() {}

export function initialize(device, data) {
    return data
}

export function toString(device, data) {
    return data.toString()
}

export function sendToMainThread(device, data) {
    return data
}

export function clone(device, data) {
    return data.constructor(data)
}

export function convert(device, data, type) {
    return type(data)
}

export function transform(device, data, mapFn) {
    return data.constructor.from(data, mapFn)
}