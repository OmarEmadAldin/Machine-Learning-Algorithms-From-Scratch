# K-Nearest Neighbors (KNN) Classifier

A clean implementation of the KNN algorithm from scratch using Python and NumPy, tested on the Iris dataset.

## Files

| File | Description |
|---|---|
| `Knn.py` | KNN classifier implementation |
| `helper_function.py` | Euclidean distance & decision boundary plotting |
| `main.py` | Training, prediction, and accuracy evaluation |
| `knn_animation.py` | Step-by-step animation of how KNN classifies a point |

## How It Works

KNN is a simple, non-parametric classification algorithm:

1. **Store** all training data (no actual "training" step)
2. **Given a new point**, compute its distance to every training point
3. **Find the K closest** neighbors
4. **Vote** — the most common class among the K neighbors wins

## Getting Started

**Install dependencies:**
```bash
pip install numpy matplotlib scikit-learn
```

**Run the classifier:**
```bash
python main.py
```

**Run the animation:**
```bash
python knn_animation.py
```

## Animation

`knn_animation.py` visually walks through the KNN process step by step:

- 🔵 Scans all training points and shows the expanding search radius
- 🟡 Highlights the K nearest neighbors with connecting lines
- 📊 Updates a live vote bar chart as each neighbor is revealed
- ✅ Shows the final prediction vs. the true label

You can change `TEST_IDX` (0–29) in the file to animate a different test point, and `K` to try different values.

## Results

Tested on the Iris dataset with an 80/20 train-test split:

```
Accuracy: ~0.97
```

## Project Structure

```
├── Knn.py
├── helper_function.py
├── main.py
└── knn_animation.py
```
