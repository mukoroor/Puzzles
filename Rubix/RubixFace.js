import RubixPiece from "./RubixPiece.js";

export default class RubixFace {
    static FACE_ADJACENCY = Object.freeze({
        0: [1, 2, 4, 3],
        1: [0, 3, 5, 2],
        2: [0, 1, 5, 4],
    });
    id;
    #lengthFunc;
    corners = [];
    edges = [];
    interiors = [];
    center = null;

    constructor(id, lengthFunc) {
        this.id = id;
        this.#lengthFunc = lengthFunc;
    }

    get length() {
        return this.#lengthFunc();
    }

    addCenter(faceCenter) {
        this.center = faceCenter;
    }

    addCorner(faceCorner) {
        this.corners.push(faceCorner);
    }

    addEdge(faceEdge) {
        this.edges.push(faceEdge);
    }

    addInterior(faceInterior) {
        this.interiors.push(faceInterior);
    }

    rotate(direction, count) {
        count = count % 4;

        console.log(this.corners)
        const nextCornerPieces = this.#nextCornerPieces(direction, count);
        console.log(nextCornerPieces);

        
        const updatedPieceData = this.corners.map((e, i) => nextCornerPieces[i].generateNextData(e, this.id));

        console.log(updatedPieceData)
        
        nextCornerPieces.forEach((e, i) => {
            const nextData = updatedPieceData[i];
            
            e.faceData = nextData;
        })
    }

    fill() {
        let max = this.length - 2;

        let rings = [this.#buildCornerRing()];

        for (let i = max; i > 0; i -= 2) {
            let ring = this.#buildRing(i);
            if (i === 1) this.center = ring;
            rings.unshift(ring);
        }

        for (let i = 1; i < rings.length; i++) {
            this.#merge(rings[i - 1], rings[i]);
        }
    }

    #buildRing(length) {
        if (length === 1) {
            this.center = new RubixPiece(`${this.id}`, `${this.id}`, "CENTER");
            return [this.center];
        }

        const maxChain = length - 1;
        const faceJoints = this.#jointPath;
        const cycle = Array(4 * maxChain);

        for (let i = 0; i < 4; i++) {
            let jointStart = faceJoints[i],
                jointEnd = 5 - jointStart;
            for (let j = 0; j < maxChain; j++) {
                let piece = new RubixPiece(`${this.id}`, `${this.id}`, "INTERIOR"),
                    prevPiece =
                        cycle[(cycle.length + maxChain * i + j - 1) % cycle.length];
                if (i || j) {
                    this.addInterior(piece);
                    RubixPiece.join(piece, prevPiece, jointStart, jointEnd);
                }

                cycle[maxChain * i + j] = piece;
            }
        }
        RubixPiece.join(
            cycle[0],
            cycle[4 * maxChain - 1],
            faceJoints[0],
            5 - faceJoints[0]
        );

        return cycle;
    }

    #buildCornerRing() {
        const joints = this.#jointPath;
        const targetCornerActivations = [this.id, joints[0], joints[1]];
        let target = this.corners.find((corner) =>
            targetCornerActivations.every(
                (activation) => typeof corner.getFace(activation) === "number"
            )
        );

        const outerChain = this.length - 1;
        const cycle = Array(4 * outerChain);

        for (let i = 0; i < 4; i++) {
            let joint = 5 - joints[i];
            for (let j = 0; j < outerChain; j++) {
                target = target.getFace(joint);
                cycle[outerChain * i + j] = target;
            }
        }

        return cycle;
    }

    #merge(innerRing, outerRing) {
        const mergeJoints = this.#jointPath;
        const innerSize = Math.floor(innerRing.length / 4) + 1;
        mergeJoints.push(mergeJoints.shift(), null);

        innerRing.unshift(innerRing.pop());

        let innerPointer = 0,
            outerPointer = 0,
            counter = 0;
        let face = mergeJoints.shift(),
            flag = false;

        while (innerPointer <= innerRing.length && face !== null) {
            counter++;
            RubixPiece.join(
                innerRing[innerPointer++ % innerRing.length],
                outerRing[outerPointer++],
                face,
                5 - face
            );
            if (!(counter % innerSize) && !flag) {
                innerPointer--;
                outerPointer++;
                face = mergeJoints.shift();
                flag = innerSize !== 1;
            } else {
                flag = false;
            }
        }
    }

    #nextCornerPieces(direction, count) {
        return direction === this.#cornerOrientation
            ? this.corners.map((_, i) => this.corners[(i + count) % 4])
            : this.corners.map((_, i) => this.corners[(i - count + 4) % 4]);
    }

    get #jointPath() {
        return this.id < 3
            ? [...RubixFace.FACE_ADJACENCY[this.id]]
            : RubixFace.FACE_ADJACENCY[5 - this.id].toReversed();
    }

    get #cornerOrientation() {
        return this.id % 3 ? "cw" : "ccw";
    }

    toString() {
        return `~~${this.id}~~
                CORNERS: ${this.corners.map((corner) => corner.toString())}\n
                EDGES: ${this.edges.map((edge) => edge.toString())}\n
                INTERIORS: ${this.interiors.map((interior) =>
            interior.toString()
        )}\n\n`;
    }
}
