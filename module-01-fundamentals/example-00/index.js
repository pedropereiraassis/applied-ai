import tf from "@tensorflow/tfjs-node";

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  // First neural network layer.
  // Input shape has 7 values (normalized age + 3 colors + 3 locations).

  // 80 neurons because we have a very small training dataset.
  // More neurons increase the complexity the model can learn,
  // but also increase compute cost.

  // ReLU acts as a filter:
  // positive values pass through, zero/negative values are blocked.

  model.add(
    tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }),
  );

  // Output layer: 3 neurons, one for each category.
  // Categories: premium, medium, basic.

  // Softmax converts outputs into probabilities.

  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  // Compile the model.
  // Adam (Adaptive Moment Estimation) is a modern optimizer for neural networks.
  // It adjusts weights efficiently by using error history.

  // Loss: categoricalCrossentropy.
  // Compares predicted class scores to the correct target.
  // Premium class target is [1, 0, 0].

  // The farther the prediction is from the correct answer, the higher the loss.
  // Typical use cases: image classification, recommendations, user categorization,
  // or any single-choice classification task.

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // Model training.
  // verbose: disables full logs (callbacks can be used instead).
  // epochs: number of full passes over the training data.
  // shuffle: randomizes data order to reduce training bias.

  await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    shuffle: true,
    callbacks: {
      // onEpochEnd: (epoch, logs) =>
      //   console.log(`Epoch ${epoch}: loss = ${logs.loss}`),
    },
  });

  return model;
}

async function predict(model, personTensor) {
  // Transform JavaScript array into a tensor.
  const tfInput = tf.tensor2d(personTensor);

  // Run prediction (output has 3 class probabilities).
  const prediction = model.predict(tfInput);
  const predictionArray = await prediction.array();
  return predictionArray[0].map((probability, index) => ({
    probability,
    index,
  }));
}

// Example training people (each person has age, color, and location).
// const people = [
//     { name: "Erick", age: 30, color: "blue", location: "Sao Paulo" },
//     { name: "Ana", age: 25, color: "red", location: "Rio" },
//     { name: "Carlos", age: 40, color: "green", location: "Curitiba" }
// ];

// Input vectors with normalized values and one-hot encoding.
// Order: [normalized_age, blue, red, green, sao_paulo, rio, curitiba]
// const peopleTensor = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// We only use numeric data because neural networks operate on numbers.
// normalizedPeopleTensor is the model's input dataset.
const normalizedPeopleTensor = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels for target categories (one-hot encoded).
// [premium, medium, basic]
const labelNames = ["premium", "medium", "basic"]; // Label order.
const labelTensor = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Create input (xs) and output (ys) tensors to train the model.
const inputXs = tf.tensor2d(normalizedPeopleTensor);
const outputYs = tf.tensor2d(labelTensor);

// More data generally helps the algorithm learn complex patterns better.
const model = await trainModel(inputXs, outputYs);

const person = { name: "jose", age: 28, color: "green", location: "Curitiba" };
// Normalize the new person's age using the same training range.

// Example: min_age = 25, max_age = 40, (28 - 25) / (40 - 25) = 0.2

const normalizedPersonTensor = [
  [
    0.2, // normalized age
    1, // blue color
    0, // red color
    0, // green color
    1, // sao_paulo location
    0, // rio location
    0, // curitiba location
  ],
];

const predictions = await predict(model, normalizedPersonTensor);
const results = predictions
  .sort((a, b) => b.probability - a.probability)
  .map(
    (prediction) =>
      `${labelNames[prediction.index]} (${(prediction.probability * 100).toFixed(2)}%)`,
  )
  .join("\n");

console.log(results);
