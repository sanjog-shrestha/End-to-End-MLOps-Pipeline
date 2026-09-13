from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import uvicorn

app = FastAPI(title="Student Risk Prediction API", description="API for predicting student risk using a trained model.", version="1.0.0")

model_pipe = joblib.load("model.pkl")

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model_pipe.predict(df)
        return {"result": prediction.tolist()[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)