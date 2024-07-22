let counter = 0;
export default class RubixPiece {
    id
    activations = {};
    type;
    faceData = Array(6);

    constructor(activationString = "012345", colorIndices = "012345", type = "CENTER") {
        this.type = type;
        this.id = counter++;
        let curr = {};
        for (let i = 0; i < activationString.length; i++) {
            curr.face = +activationString[i];
            this.activations[curr.face] = curr;
            this.setFace(curr.face, +colorIndices[i]);
            if (i + 1 === activationString.length) curr.next = this.activations[activationString[0]];
            else curr.next = {};
            curr = curr.next;
        }
    }

    setFace(index, value) {
        this.faceData[index] = value;
    }

    getFace(index) {
        return this.faceData[index];
    }

    toString() {
        return `\n${this.id} * ${this.type} , ${this.faceData.map(e =>  e.type !== undefined ? `${e.id}_${e.type}`: e).join(" , ")}`
    }

    static join(pieceA, pieceB, faceA, faceB) {
        pieceA.setFace(faceA, pieceB);
        pieceB.setFace(faceB, pieceA);
    }
}