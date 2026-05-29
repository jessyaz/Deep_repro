# Deep_repro

Industrial tools used to train and deploy machine learning models are valued for their reliability, flexibility, and ease of use. Drawing from our experience working with large, constantly evolving codebases (particularly in deep learning research projects) we would like to share a simple yet effective setup for making experiments both traceable and reproducible.

This repo is a more intricated example of reproducibility in deep learning, a standard example using Linear Regression is available in the [Linear_regression_reproducibility](https://github.com/J-ally/Deep_repro_LR) repository.

Slides for the presentation are available in the /slides folder.


# Installing everyting

To install the required dependencies, you can use the following command:

```bash
uv sync
```

Then you need to download the dataset :
```bash
uv run dataops/pretraitment.py
```

# Running the code
To run the code, you can use the following command:

```bash
uv run train.py
```

# Inspecting the results

To inspect the results, you can use the following command:

```bash
mlflow ui
```

If there is already a process using the port and you can stop it

```bash
lsof -ti :5000 | xargs kill -9
```

# Evaluating a model

To evaluate a model, you can use the following command (obtaining the run_id from the mlflow ui in the artifacts section of the run) :

```bash
uv run evaluate.py run_id=<run_id>
```

## A note on the dataset label description
Each example is assigned to one of the following labels:
- 0 T-shirt/top
- 1 Trouser
- 2 Pullover
- 3 Dress
- 4 Coat
- 5 Sandal
- 6 Shirt
- 7 Sneaker
- 8 Bag
- 9 Ankle boot
