const WGSLActivations = {
    activations: new Map(),

    addActivation(name, activationOperationsWGSL, activationDerivativeOperationsWGSL) {
        if (this.activations.has(name)) {
            throw new Error(`Activation '${name}' already exists.`);
        };

        const obj = {
            id: this.activations.size + 1,
            name: name,
            activationOperationsWGSL: activationOperationsWGSL,
            activationDerivativeOperationsWGSL: activationDerivativeOperationsWGSL,
        };
        this.activations.set(name, obj);
    },

    activationFunctionName(containerName) {
        return `activation_${containerName}`;
    },

    activationDerivativeFunctionName(containerName) {
        return `activation_derivative_${containerName}`;
    },

    createFunctionBody(operationsWGSL) {
        if (typeof operationsWGSL === 'string') {
            return /*wgsl*/`return ${operationsWGSL};`
        } else if (operationsWGSL instanceof Array) {
            return (operationsWGSL.length > 1 ? 
                /*wgsl*/`${operationsWGSL?.slice(0, -1).map(l => l.endsWith(';') ? l : l + ';').join('\n')}
                ` : '') +  /*wgsl*/`return ${operationsWGSL.at(-1)};`
        }
        return undefined;
    },

    createActivationFunction(obj) {
        return /*wgsl*/`
            fn ${this.activationFunctionName(obj.name)}(${X_VAR}: f32) -> f32 {
                ${this.createFunctionBody(obj.activationOperationsWGSL) || ONE_LITERAL}
            }
        `;
    },

    createActivationDerivativeFunction(obj) {
        return /*wgsl*/`
            fn ${this.activationDerivativeFunctionName(obj.name)}(${X_VAR}: f32) -> f32 {
                ${this.createFunctionBody(obj.activationDerivativeOperationsWGSL) || ZERO_LITERAL}
            }
        `;
    },

    createActivationFunctionCall(name) {
        return this.createFunctionCall(this.activationFunctionName(name), X_VAR);
    },

    createActivationDerivativeFunctionCall(name) {
        return this.createFunctionCall(this.activationDerivativeFunctionName(name), X_VAR);
    },

    createFunctionCall(name, vars) {
        return `${name}(${typeof vars === 'string' ? vars: vars.join()})`
    },

    createSwitch(id, operationsWGSL) {
        return /*wgsl*/`
            case ${id}: {
                ${this.createFunctionBody(operationsWGSL)}
            }`
    },

    createActivationSwitchCase(obj) {
        return this.createSwitch(obj.id, this.createActivationFunctionCall(obj.name));
    },

    createActivationDerivativeSwitchCase(obj) {
        return this.createSwitch(obj.id, this.createActivationDerivativeFunctionCall(obj.name));
    },

    createActivationShaderCode() {
        const entries = Array.from(this.activations.entries());

        const activations = /*wgsl*/`${entries.map(([_, actObj]) => this.createActivationFunction(actObj)).join('\n')}`

        const derivatives = /*wgsl*/`${entries.map(([_, actObj]) => this.createActivationDerivativeFunction(actObj)).join('\n')}`
        
        const activationSwitch = /*wgsl*/`${entries.map(([_, actObj]) => this.createActivationSwitchCase(actObj)).join('\n')}`

        const derivativeSwitch = /*wgsl*/`${entries.map(([_, actObj]) => this.createActivationDerivativeSwitchCase(actObj)).join('\n')}`

        return { activations, derivatives, activationSwitch, derivativeSwitch };
    },

    getActivationId(name) {
        return this.activations.get(name)?.id || Number.MAX_SAFE_INTEGER;
    },

    getActivations() {
        return Array.from(this.activations.keys())
    }
}
export default WGSLActivations;

export const X_VAR = /*wgsl*/`x`;
const ZERO_LITERAL = /*wgsl*/`0f`;
const ONE_LITERAL = /*wgsl*/`1f`;

const SIN = /*wgsl*/`sin(${X_VAR})`
const COS = /*wgsl*/`cos(${X_VAR})`

const POS_EXP = /*wgsl*/`exp(${X_VAR})`
const POS_EXP_NAME = 'posExp';
const POS_EXP_VAR = /*wgsl*/`var ${POS_EXP_NAME} = ${POS_EXP}`;

const NEG_EXP = /*wgsl*/`exp(-${X_VAR})`
const NEG_EXP_NAME = 'negExp';
const NEG_EXP_VAR = /*wgsl*/`var ${NEG_EXP_NAME} = ${NEG_EXP}`;

const SIG = /*wgsl*/`1 / (1 + ${NEG_EXP})`;

const TANH_ACT = [POS_EXP_VAR, NEG_EXP_VAR, /*wgsl*/`(posExp - negExp) / (posExp + negExp)`];

const ALPHA = 0.2;

WGSLActivations.addActivation('DROPOUT', ZERO_LITERAL, ZERO_LITERAL);
WGSLActivations.addActivation('CONSTANT', ONE_LITERAL, ZERO_LITERAL);
WGSLActivations.addActivation('LINEAR', X_VAR, ONE_LITERAL);
WGSLActivations.addActivation('SIN', SIN, COS);
WGSLActivations.addActivation('COS', COS, `-${SIN}`);
WGSLActivations.addActivation('GAUSSIAN', /*wgsl*/`exp(-pow(${X_VAR}, 2))`, /*wgsl*/`-2 * ${X_VAR} * ${WGSLActivations.createActivationFunctionCall('GAUSSIAN')}`);
WGSLActivations.addActivation('SIGMOID', SIG, [/*wgsl*/`var sig = ${WGSLActivations.createActivationFunctionCall('SIGMOID')}`, /*wgsl*/`sig * (1 - sig)`]);
WGSLActivations.addActivation('TANH', TANH_ACT, [/*wgsl*/`var tanh = ${WGSLActivations.createActivationFunctionCall('TANH')}`, /*wgsl*/`1  - pow(tanh, 2)`]);
WGSLActivations.addActivation('BINARY_STEP',  /*wgsl*/`select(${ONE_LITERAL}, ${ZERO_LITERAL}, ${X_VAR} < 0)`, ZERO_LITERAL);
WGSLActivations.addActivation('SOFTPLUS', /*wgsl*/`log(1 + ${POS_EXP})`, SIG);
WGSLActivations.addActivation('SiLU', /*wgsl*/`${X_VAR} * ${SIG}`,  /*wgsl*/`(1 + (${X_VAR} + 1) * ${NEG_EXP}) / pow(1 + ${NEG_EXP}, 2)`);
WGSLActivations.addActivation('ReLU', /*wgsl*/`max(${ZERO_LITERAL}, ${X_VAR})`, /*wgsl*/`select(${ZERO_LITERAL}, ${ONE_LITERAL}, ${X_VAR} > 0)`);
WGSLActivations.addActivation('LeakyReLU', /*wgsl*/`select(0.01, 1, ${X_VAR} > 0) * ${X_VAR}`, /*wgsl*/`select(select(0.01, 0, ${X_VAR} == 0), ${ONE_LITERAL}, ${X_VAR} > 0)`);
WGSLActivations.addActivation('ParReLU', /*wgsl*/`select(${ALPHA}, 1, ${X_VAR} > 0) * ${X_VAR}`, /*wgsl*/`select(select(${ALPHA}, 0, ${X_VAR} == 0), ${ONE_LITERAL}, ${X_VAR} > 0)`);


