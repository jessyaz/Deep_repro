# Deep_repro
Industrial tools used to train and deploy machine learning models are valued for their reliability, flexibility, and ease of use. From our experience working with large, constantly evolving codebases—particularly in the context of deep learning research projects—we’d like to share a simple and effective setup that makes experiments reproducible.


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
lsof -ti :5000 | xargs kill -9
mlflow ui
```

# Evaluating a model

To evaluate a model, you can use the following command (obtaining the run_id from the mlflow ui in the artifacts section of the run) :

```bash
uv run evaluate.py run_id=<run_id>
```
