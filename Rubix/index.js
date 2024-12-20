import RubixPuzzleDrawer from "./RubixPuzzleDrawer.js";
import NeuralNetwork from "../NeuralNetwork/NeuralNetwork.js";

const d = new RubixPuzzleDrawer();

// test 1
// const testNetwork = new NeuralNetwork([3, 2, 1]);
// const X = [
// //   [0.1, 0.3, 0.7],
// //   [0.2, 0.8, 0.6],
// //   [0.4, 0.4, 0.9],
// //   [0.3, 0.5, 0.2],
// //   [0.7, 0.1, 0.8],
// //   [0.6, 0.6, 0.4],
// //   [0.8, 0.2, 0.5],
// //   [0.5, 0.7, 0.3],
// //   [0.2, 0.6, 0.1],
// //   [0.1, 0.2, 0.6],
// //   [0.9, 0.3, 0.7],
// //   [0.3, 0.9, 0.4],
// //   [0.7, 0.8, 0.2],
// //   [0.8, 0.4, 0.6],
// //   [0.6, 0.5, 0.7],
// //   [0.4, 0.3, 0.9],
// //   [0.5, 0.9, 0.2],
// //   [0.9, 0.1, 0.3],
// //   [0.2, 0.4, 0.7],
// //   [0.7, 0.5, 0.1],
//   [  1,   1,   1],   
// ]
// const y = [
// //   [0], [1], [0],
// //   [1], [0], [1],
// //   [0], [1], [0],
// //   [1], [0], [1],
// //   [0], [1], [0],
// //   [1], [0], [1],
// //   [0], [1],
//   [1],
// ]

// test 2
// const testNetwork = new NeuralNetwork([3, 2, 2]);
// let X = [
// [0.2,0.1,0.7],
// [0.6,0.1,0.3],
// [0.4,0.3,0.3],
// [0, 0.8, 0.2],
// ]

// let y = [
// [1,0],
// [0,1],
// [1,0],
// [0,1]
// ]

// test 3
// const inDim = 256;
// const outDim = 10;
// const testNetwork = new NeuralNetwork([inDim, 128, 64, outDim]);
// let l = 21;
// let X = Array.from({length: l}, () => {
//   return Array.from({length: inDim}, () => Math.random() * 2 - 1)
// })

// let y = Array.from({length: l}, () => {
//   let vec  = Array.from({length: outDim}, () => Math.random() * 2 - 1)
//   // vec[Math.floor(Math.random() * outDim)] = 1;
//   return vec;
// })

//test 4
const testNetwork = new NeuralNetwork([2, 2, 1]);
let X = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
];
let y = [
  [0],
  [1],
  [1],
  [0]
];

//test 5
// const testNetwork = new NeuralNetwork([2, 2, 1]);
// let X = [
//   [0.35, 0.9],
// ];
// let y = [
//   [0.5],
// ];

//test 6
// const testNetwork = new NeuralNetwork([1, 2, 1]);
// let X = [
// [1], 
// [2], 
// // [4]
// ]
// let y = [
//     [0.2], 
//     [0.4], 
//     // [8]
// ]

    

document.addEventListener("DOMContentLoaded", async () =>  {
    // d.draw();

    // let request = await fetch('./testNumbers.json')
    // let response = await request.json()
    
    // const testNetwork = new NeuralNetwork([784, 16, 16, 10]);

    // const XT = [];
    // const YT = [];
    // const XTE = [];
    // const YTE = [];
    // for (let i = 0; i < 80; i++) {
    //     const key = Object.keys(response[i])[0];
    //     // if (XT.length == ) break;
    //     if (i % 5 != 0) {
    //         XT.push(response[i][key]);
    //         let arr = Array(10).fill(0);
    //         arr[key] = 1;
    //         YT.push(arr);
    //     } else {
    //         XTE.push(response[i][key]);
    //         let arr = Array(10).fill(0);
    //         arr[key] = 1;
    //         YTE.push(arr);
    //     }
    // }
    // console.log(XT, YT)
    // console.log(XTE, YTE)
    // let yPred = await testNetwork.predict(XTE)
    // console.log(yPred)
    // console.log('TEST ERROR', NeuralNetwork.meanSquaredError(YTE, yPred))
    // yPred = await testNetwork.predict(XT)
    // // console.log(await testNetwork.extractNetworkParameters())

    // console.log(yPred)
    // console.log('TRAIN ERROR', NeuralNetwork.meanSquaredError(YT, yPred))

    // await testNetwork.train(XT, YT, 1000, 0.5, 16);
    // // console.log(await testNetwork.extractNetworkParameters())

    // yPred = await testNetwork.predict(XTE)
    // console.log(yPred)
    // console.log('TEST ERROR', NeuralNetwork.meanSquaredError(YTE, yPred));

    // yPred = await testNetwork.predict(XT)
    // console.log(yPred)
    // console.log('TRAIN ERROR', NeuralNetwork.meanSquaredError(YT, yPred));
    
    let yPred = await testNetwork.predict(X)
    console.log(yPred)
    console.log('TRAIN ERROR: START', NeuralNetwork.meanSquaredError(y, yPred));

    console.log(await testNetwork.extractNetworkParameters())
    await testNetwork.train(X, y, 2000, 1, 4);
    console.log(await testNetwork.extractNetworkParameters())

    yPred = await testNetwork.predict(X)
    console.log(y, yPred)
    console.log('TRAIN ERROR: END', NeuralNetwork.meanSquaredError(y, yPred));
});


