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
  static edgeCycles = [
    "10",
    "03",
    "31",
    "20",
    "12",
    "04",
    "24",
    "43",
    "45",
    "35",
    "52",
    "51",
  ];

  length;
  faces;
  moves = [];

  constructor(length) {
    this.length = length;
    this.faces = Array.from(
      { length: 6 },
      (_, i) => new RubixFace(i, () => length)
    );

    if (length === 1) {
      const onlyCube = new RubixPiece();
      this.faces.forEach((face) => face.addCenter(onlyCube));
      return;
    }

    this.buildEdges(this.buildCorners());
    this.buildFaces();
  }

  buildFaces() {
    this.faces.forEach((face) => face.fill());
  }

  buildCorners() {
    const corners = [];
    for (const cycle of RubixPuzzle.cornerCycles) {
      const corner = new RubixPiece(cycle, cycle, "CORNER");
      cycle.split("").forEach((faceId) => {
        this.faces[faceId].addCorner(corner);
      });
      corners.push(corner);
    }
    return corners;
  }

  buildEdges(corners) {
    let edgeCycleIndex = 0;
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
        const cycle_color_Str = RubixPuzzle.edgeCycles[edgeCycleIndex++];
        for (let j = 0; j < this.length - 2; j++) {
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
          RubixPiece.join(adjacentArray[j], adjacentArray[j - 1], ...singular);
        }
      }
    }
  }

  get pieceCount() {
    return Math.pow(this.length, 3) - Math.pow(Math.max(this.length - 2, 0), 3);
  }

  reverseLastMove() {
    if (this.moves.length == 0) return undefined
    return this.#reverseMove(this.moves.pop())

  }

  #reverseMove([face, depth, direction, count]) {
    return [face, depth, `${(direction.length > 2 ? "": "c")}cw`, count, false]
  }

  reset() {}

  rotate(face, depth = 0, direction = "cw", count = 0,  store=true) {
    if (this.length === 1) return;
    if (store) this.moves.push([face, depth, direction, count]);
    this.faces[face].rotate(depth, direction, count);
  }

  scramble(scrambleCount = SCRAMBLE_MOVES, withLog = true) {
    if (this.length === 1) return;
    console.time("scramble");
    let face, depth, count, direction;
    const scores = []
    for (let i = 0; i < scrambleCount; i++) {
      face = Math.floor(Math.random() * 6);
      depth = Math.floor(Math.random() * (this.length - 1));
      direction = Math.random() > 0.5 ? "cw" : "ccw";
      count = Math.floor(Math.random() * 3) + 1;
      // scores.push(this.scores());
      this.rotate(face, depth, direction, count);
    }
    if (withLog) {
      console.log(
        "%cThese are the moves to fix the cube",
        "font-weight: bold; color: orange; font-size: 1.25em;"
      );
      console.table(
        [["Face", "Depth", "Direction", "Count"]].concat(
          this.moves.toReversed().map(e => this.#reverseMove(e))
        )
      );
      // console.log(scores);
    }
    console.timeEnd("scramble");

  }

  crossPattern() {
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < this.length; j += 2) {
          this.moves.push([i, j, "cw", 2]);
          this.faces[i].rotate(j, "cw", 2);
      }
    }
  }

  framePattern() {
    for (let i = 0; i < 3; i++) {
      this.faces[i].rotate({ count: 2 });
      this.faces[5 - i].rotate({ count: 2 });
    }
  }

  toString() {
    return this.faces.map((face) => face.toString()).join("\n");
  }

  scores() {
    return this.faces.map(e => e.scoreDP());
    // return this.faces.map(e => [e.score(), e.scoreDFS(), e.scoreDP()]);
  }
}

const SCRAMBLE_MOVES = 20;

// console.time('testInit')
// const testRubix = new RubixPuzzle(5);
// // testRubix.scramble(SCRAMBLE_MOVES, false)
// // console.log(testRubix.scores())
// // testRubix.faces.forEach((e, i) => {
// //   const twoD = e.to2DArray().map(x => x.map(c => c.faceData[e.id]));
// //   console.log(i)
// //   console.table(twoD)
// // })

// const g = [
//   [1, 1, 1, 2, 0],
//   [1, 0, 1, 2, 2],
//   [1, 1, 1, 1, 1],
//   [4, 4, 3, 2, 1],
//   [4, 4, 5, 5, 1]
// ]

// testRubix.faces[0].scoreDP(g);


// console.timeEnd('testInit')
