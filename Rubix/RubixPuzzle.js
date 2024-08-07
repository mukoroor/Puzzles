import RubixFace from "./RubixFace.js";
import RubixPiece from "./RubixPiece.js";

export default class RubixPuzzle {
    static cornerCycles = [
        "031",
        "012",
        "024",
        "043",
        "534",
        "542",
        "521",
        "513",
    ];

    length;
    faces;
    moves;

    constructor(length) {
        this.length = length;
        this.faces = Array.from({ length: 6 }, (_, i) => new RubixFace(i, () => length));

        if (length === 1) {
            const onlyCube = new RubixPiece();
            this.faces.forEach((face) => face.addCenter(onlyCube));
            return;
        }

        this.buildEdges(this.buildCorners());
        this.buildFaces();
    }

    buildFaces() {
        this.faces.forEach(face => face.fill())
    }

    buildCorners() {
        const corners = []
        for (const cycle of RubixPuzzle.cornerCycles) {
            const corner = new RubixPiece(cycle, cycle, "CORNER");
            cycle
            .split("")
            .forEach((faceId) => {
                this.faces[faceId].addCorner(corner)
            });
            corners.push(corner)
        }
        return corners;
    }

    buildEdges(corners) {
        for (let i = 0; i < RubixPuzzle.cornerCycles.length; i++) {
            for (let k = i + 1; k < RubixPuzzle.cornerCycles.length; k++) {
                const setCornerI = new Set(RubixPuzzle.cornerCycles[i].split(""));
                const setCornerK = new Set(RubixPuzzle.cornerCycles[k].split(""));

                const inBoth = [],
                    singular = [];

                for (const char of setCornerI) {
                    if (setCornerK.has(char)) {
                        inBoth.push(char);
                        setCornerI.delete(char);
                        setCornerK.delete(char);
                    }
                }

                if (inBoth.length !== 2) continue;
                singular.push(...setCornerI, ...setCornerK);

                const adjacentArray = [corners[i]];
                for (let j = 0; j < this.length - 2; j++) {
                    const cycle_color_Str = inBoth.join("");
                    const edgePiece = new RubixPiece(
                        cycle_color_Str,
                        cycle_color_Str,
                        "EDGE"
                    );
                    adjacentArray.push(edgePiece);
                    inBoth.forEach((faceId) => this.faces[faceId].addEdge(edgePiece));
                }

                adjacentArray.push(corners[k]);
                for (let j = 1; j < adjacentArray.length; j++) {
                    RubixPiece.join(adjacentArray[j], adjacentArray[j - 1], ...singular)
                }
            }
        }
    }

    get pieceCount() {
        return Math.pow(this.length, 3) - Math.pow(Math.max(this.length - 2, 0), 3);
    }

    updateColors(newColors) { }

    undo() { }

    reset() { }

    scramble() { }

    toString() {
        return this.faces.map(face => face.toString()).join('\n')
    }
}

const p = new RubixPuzzle(1);

// console.log(p.faces[0].corners[0].activations)
