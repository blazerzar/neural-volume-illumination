const IMAGE_SIZE = 28 * 28;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const TRAIN_TEST_RATIO = 5 / 6;
const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE =
    "https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png";
const MNIST_LABELS =
    "https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8";

async function main() {
    tf.setBackend("webgpu").then(async () => {
        const data = new MnistData();
        await data.load();

        const model = tf.sequential();
        model.add(
            tf.layers.dense({
                inputShape: [784],
                units: 64,
                activation: "relu",
            }),
        );
        model.add(tf.layers.dense({ units: 64, activation: "relu" }));
        model.add(tf.layers.dense({ units: 64, activation: "relu" }));
        model.add(tf.layers.dense({ units: 64, activation: "relu" }));
        model.add(tf.layers.dense({ units: 64, activation: "relu" }));
        model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

        model.compile({
            optimizer: "adam",
            loss: "categoricalCrossentropy",
            metrics: ["accuracy"],
        });

        const { xs, ys } = data.getTrainData();

        console.log("Data ready");
        let prevTime = performance.now();

        await model.fit(xs, ys, {
            epochs: 5,
            batchSize: 64,
            verbose: 1,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    const time = (performance.now() - prevTime) / 1000;
                    console.log(
                        `Epoch ${epoch + 1}: ` +
                            `loss=${logs.loss.toFixed(4)}, ` +
                            `acc=${logs.acc?.toFixed(4) ?? logs.accuracy.toFixed(4)} ` +
                            `(${time} s)`,
                    );
                    prevTime = performance.now();
                },
            },
        });

        console.log("Done");
    });
}

window.onload = () => main();

class MnistData {
    async load() {
        // Load sprite image
        const img = new Image();
        img.crossOrigin = "";
        const imgLoaded = new Promise((resolve) => {
            img.onload = () => resolve();
        });
        img.src = MNIST_IMAGES_SPRITE;

        // Load labels
        const labelsRequest = fetch(MNIST_LABELS);

        await Promise.all([imgLoaded, labelsRequest]);
        const labels = new Uint8Array(
            await (await labelsRequest).arrayBuffer(),
        );

        // Allocate buffers
        this.datasetImages = new Float32Array(
            NUM_DATASET_ELEMENTS * IMAGE_SIZE,
        );
        this.datasetLabels = labels;

        // Chunked canvas processing
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        const chunkSize = 1000; // images per chunk
        canvas.width = img.width;
        canvas.height = chunkSize * 28;

        let pixelIndex = 0;

        for (let i = 0; i < NUM_DATASET_ELEMENTS; i += chunkSize) {
            const currentChunkSize = Math.min(
                chunkSize,
                NUM_DATASET_ELEMENTS - i,
            );

            ctx.clearRect(0, 0, canvas.width, canvas.height);

            ctx.drawImage(
                img,
                0,
                i * 28, // source x, y
                img.width,
                currentChunkSize * 28, // source w, h
                0,
                0, // dest x, y
                img.width,
                currentChunkSize * 28, // dest w, h
            );

            const imageData = ctx.getImageData(
                0,
                0,
                canvas.width,
                currentChunkSize * 28,
            );

            for (let j = 0; j < imageData.data.length; j += 4) {
                this.datasetImages[pixelIndex++] = imageData.data[j] / 255;
            }
        }

        // Split train / test
        this.trainImages = this.datasetImages.slice(
            0,
            IMAGE_SIZE * NUM_TRAIN_ELEMENTS,
        );
        this.testImages = this.datasetImages.slice(
            IMAGE_SIZE * NUM_TRAIN_ELEMENTS,
        );

        this.trainLabels = this.datasetLabels.slice(
            0,
            NUM_CLASSES * NUM_TRAIN_ELEMENTS,
        );
        this.testLabels = this.datasetLabels.slice(
            NUM_CLASSES * NUM_TRAIN_ELEMENTS,
        );
    }

    getTrainData() {
        return {
            xs: tf.tensor2d(this.trainImages, [NUM_TRAIN_ELEMENTS, IMAGE_SIZE]),
            ys: tf.tensor2d(this.trainLabels, [
                NUM_TRAIN_ELEMENTS,
                NUM_CLASSES,
            ]),
        };
    }

    getTestData() {
        return {
            xs: tf.tensor2d(this.testImages, [NUM_TEST_ELEMENTS, IMAGE_SIZE]),
            ys: tf.tensor2d(this.testLabels, [NUM_TEST_ELEMENTS, NUM_CLASSES]),
        };
    }
}
