from fastapi import FastAPI
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from pydantic import BaseModel

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Load Iris dataset
iris = load_iris()

# Extract features and labels
X, y = iris.data, iris.target

# Train the model
clf = GaussianNB()
clf.fit(X, y)

# Create FastAPI instance
app = FastAPI()

# Define prediction endpoint
@app.post("/predict")
def predict(data: IrisFeatures):
    test_data = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    class_idx = clf.predict(test_data)[0]
    return {"class": iris.target_names[class_idx]}