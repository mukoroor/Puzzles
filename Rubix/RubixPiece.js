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
            if (!i) this.activations.start = curr;
            if (i + 1 === activationString.length) curr.next = this.activations[activationString[0]];
            else curr.next = {};
            curr = curr.next;
        }
    }

    generateNextData(mimicPiece, constFace) {
        const constFaceExists = !!this.activations[constFace];
        let marker = false;
        const faceData = [...this.faceData];
        let curr = this.activations[constFace] || this.activations.start;
        let currMimic = mimicPiece.activations[constFace] || mimicPiece.activations.start;

        if (!constFaceExists && curr.face === currMimic.face) curr = curr.next;


        while (!marker || (constFaceExists && curr.face !== constFace) || (!constFaceExists && currMimic.face !== mimicPiece.activations.start.face)) {
            marker = true;
            faceData[curr.face] = mimicPiece.getFace(currMimic.face);
            curr = curr.next;
            currMimic = currMimic.next;
        }

        return faceData
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