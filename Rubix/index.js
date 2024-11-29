import RubixPuzzleDrawer from "./RubixPuzzleDrawer.js";
import NeuralNetwork from "../NeuralNetwork/NeuralNetwork.js";

const d = new RubixPuzzleDrawer();
const testNetwork = new NeuralNetwork([3, 2, 2, 1]);

const X = [
  [0.1, 0.3, 0.7],
  [0.2, 0.8, 0.6],
  [0.4, 0.4, 0.9],
  [0.3, 0.5, 0.2],
  [0.7, 0.1, 0.8],
  [0.6, 0.6, 0.4],
  [0.8, 0.2, 0.5],
  [0.5, 0.7, 0.3],
  [0.2, 0.6, 0.1],
  [0.1, 0.2, 0.6],
  [0.9, 0.3, 0.7],
  [0.3, 0.9, 0.4],
  [0.7, 0.8, 0.2],
  [0.8, 0.4, 0.6],
  [0.6, 0.5, 0.7],
  [0.4, 0.3, 0.9],
  [0.5, 0.9, 0.2],
  [0.9, 0.1, 0.3],
  [0.2, 0.4, 0.7],
  [0.7, 0.5, 0.1],
  [  1,   1,   1],   
]
const y = [
  [0], [1], [0],
  [1], [0], [1],
  [0], [1], [0],
  [1], [0], [1],
  [0], [1], [0],
  [1], [0], [1],
  [0], [1], [1],
]

document.addEventListener("DOMContentLoaded", async () =>  {
    d.draw();

    let yPred = await testNetwork.predict(X)
    console.log('ERROR', NeuralNetwork.meanSquaredError(y, yPred));
    await testNetwork.train(X, y, 30000, 0.07, 7)
    await testNetwork.device.queue.onSubmittedWorkDone()
    yPred = await testNetwork.predict(X)
    console.log(yPred, y)
    console.log('ERROR', NeuralNetwork.meanSquaredError(y, yPred));
});


// fetch('./testNumbers.json').then(res => res.json())
// .then(async(data) => {
//     const XT = [];
//     const YT = [];
//     const XTE = [];
//     const YTE = [];
//     for (const value of data) {
//       const key = Object.keys(value)[0];
//       if (XT.length == 2) break;
//         if (Math.random() < 0.5) {
//           XT.push(value[key]);
//           let arr = Array(10).fill(0);
//           arr[key] = 1;
//           YT.push(arr);
//         } else {
//           XTE.push(value[key]);
//           let arr = Array(10).fill(0);
//           arr[key] = 1;
//           YTE.push(arr);
//         }
//     }
//     console.log(XT, YT)
//     console.log(XTE, YTE)
//     await test.train(XT, YT, 1000, 0.07, 2);
//     await test.device.queue.onSubmittedWorkDone();
//     console.timeEnd('train')
//     let yPred = await test.predict(XTE)
//     console.log(YTE, yPred)
//     console.log('ERROR', NeuralNetwork.meanSquaredError(YTE, yPred));

// })
