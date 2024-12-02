import RubixPuzzleDrawer from "./RubixPuzzleDrawer.js";
import NeuralNetwork from "../NeuralNetwork/NeuralNetwork.js";

const d = new RubixPuzzleDrawer();
// const testNetwork = new NeuralNetwork([3, 2, 1]);
// const testNetwork = new NeuralNetwork([256, 128, 64, 10]);
const testNetwork = new NeuralNetwork([3, 2, 3, 2]);

// const X = [
//   [0.1, 0.3, 0.7],
//   [0.2, 0.8, 0.6],
//   [0.4, 0.4, 0.9],
//   [0.3, 0.5, 0.2],
//   [0.7, 0.1, 0.8],
//   [0.6, 0.6, 0.4],
//   [0.8, 0.2, 0.5],
//   [0.5, 0.7, 0.3],
//   [0.2, 0.6, 0.1],
//   [0.1, 0.2, 0.6],
//   [0.9, 0.3, 0.7],
//   [0.3, 0.9, 0.4],
//   [0.7, 0.8, 0.2],
//   [0.8, 0.4, 0.6],
//   [0.6, 0.5, 0.7],
//   [0.4, 0.3, 0.9],
//   [0.5, 0.9, 0.2],
//   [0.9, 0.1, 0.3],
//   [0.2, 0.4, 0.7],
//   [0.7, 0.5, 0.1],
//   [  1,   1,   1],   
// ]
// const y = [
//   [0], [1], [0],
//   [1], [0], [1],
//   [0], [1], [0],
//   [1], [0], [1],
//   [0], [1], [0],
//   [1], [0], [1],
//   [0], [1],
//   [1],
// ]

let X = Array.from({length: 21}, () => {
  return Float32Array.from({length: 256}, () => 13 * Math.random())
})


let y = Array.from({length: 21}, () => {

  let vec  = Uint32Array.from({length: 10}, () => 0)
  vec[Math.floor(Math.random() * 10)] = 1;
  return vec;
})

y = [
[1,0],
[0,1],
[1,0],
[0,1]
]

X = 
[[0.2,0.1,0.7],
[0.6,0.1,0.3],
[0.4,0.3,0.3],
[0, 0.8, 0.2],]

document.addEventListener("DOMContentLoaded", async () =>  {
    d.draw();

    // let request = await fetch('./testNumbers.json')
    // let response = await request.json()
    
    
    // const XT = [];
    // const YT = [];
    // const XTE = [];
    // const YTE = [];
    // for (let i = 0; i < 2; i++) {
    //     const key = Object.keys(response[i])[0];
    //     // if (XT.length == ) break;
    //     if (i % 2) {
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
    // console.log(XT.length, YT)
    // console.log(XTE.length, YTE)
    // let yPred = await testNetwork.predict(XTE)
    // console.log(yPred)
    // console.log('TestError', NeuralNetwork.meanSquaredError(YTE, yPred))
    // yPred = await testNetwork.predict(XT)
    // console.log(yPred)
    // console.log('TrainError', NeuralNetwork.meanSquaredError(YT, yPred))

    // await testNetwork.train(XT, YT, 10000, 0.7);
    // yPred = await testNetwork.predict(XTE)
    // console.log(yPred)
    // console.log('TESTERROR', NeuralNetwork.meanSquaredError(YTE, yPred));
    // yPred = await testNetwork.predict(XT)
    // console.log(yPred)
    // console.log('TRAINERROR', NeuralNetwork.meanSquaredError(YT, yPred));
    
    let yPred = await testNetwork.predict(X)
    console.log(yPred)
    await testNetwork.train(X, y, 5000, 1, 4)
    console.log(await testNetwork.extractNetworkParameters())
    console.log('ERROR', NeuralNetwork.meanSquaredError(y, yPred));
    // await testNetwork.device.queue.onSubmittedWorkDone()
    yPred = await testNetwork.predict(X)
    console.log(yPred, y)
    console.log('ERROR', NeuralNetwork.meanSquaredError(y, yPred));
});


