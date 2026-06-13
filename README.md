# Mushroom Classification

A machine learning web app that predicts whether a mushroom is **edible or poisonous** based on its physical characteristics.

## Live Demo
🔗 [mushroom-classifier.up.railway.app](#)

## About
Given 12 physical features of a mushroom (cap shape, odor, gill color, etc.), the model classifies it as edible or poisonous. Built with a **Support Vector Classifier (SVC)** trained on the UCI Mushroom dataset.

### Model Performance
- Algorithm: Support Vector Classifier (SVC)
- Training dataset: UCI Mushroom Dataset (toy subset)
- Features used: 12 categorical features (label-encoded)

## Features Used for Prediction
| Feature | Example Values |
|---|---|
| Cap Shape | Bell, Conical, Flat, Knobbed, Sunken, Convex |
| Cap Surface | Fibrous, Grooves, Smooth, Scaly |
| Cap Color | Buff, Cinnamon, Red, Gray, Brown, etc. |
| Bruises | Yes / No |
| Odor | Almond, Foul, Anise, Musty, None, etc. |
| Gill Size | Broad / Narrow |
| Gill Color | Buff, Red, Gray, Chocolate, etc. |
| Stalk Root | Missing, Bulbous, Club, Equal, Rooted |
| Stalk Surface Below Ring | Fibrous, Silky, Smooth, Scaly |
| Ring Number | None, One, Two |
| Ring Type | Evanescent, Flaring, Large, None, Pendant |
| Spore Print Color | Buff, Chocolate, Black, Brown, etc. |

## Tech Stack
- **Backend:** Python, Flask
- **ML:** scikit-learn (SVC), joblib, pandas, numpy
- **Frontend:** HTML, CSS
- **Deployment:** Railway

## Run Locally
open bash and run these commands

git clone https://github.com/Fatima-I/mushroom-classifier

cd mushroom-classifier

pip install -r requirements.txt

python app.py

Then open `http://localhost:5000`
