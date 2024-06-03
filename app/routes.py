from app import app
from flask import render_template, jsonify, request
import os
import pandas as pd 
import json
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

DATASETS_DIR = 'datasets'
FEATURE_COLUMNS_FILE = 'features.json'
SELECTED_MODEL_FILE = 'selected_model.json'

models = {
        "LogisticRegression" : {
            "type": "Classification", 
            "hyperparameters": {
                "C": 1.0, 
                "solver": "lbfgs"
            }
        },
        "SVC": {
            "type": "Classification",
            "description": "SVC is a supervised learning algorithm used for classification. It finds an optimal hyperplane in an N-dimensional space (where N is the number of features) that separates the classes.",
            "hyperparameters": {
                "C": "float (default=1.0)",
                "kernel": "str (default='rbf')",
                "gamma": "float (default='scale')",
            }
        },
        "MLPClassifier": {
            "type": "Classification",
            "description": "MLPClassifier is a type of artificial neural network known as a multi-layer perceptron. It can be used for classification and regression tasks.",
            "hyperparameters": {
                "hidden_layer_sizes": "tuple (default=(100,))",
                "activation": "str (default='relu')",
                "solver": "str (default='adam')",
                "alpha": "float (default=0.0001)",
            }
        },
        "GaussianNB": {
            "type": "Classification",
            "description": "GaussianNB is a simple classification algorithm based on Bayes' theorem with a Gaussian naive assumption. It assumes that features are independent and follow a normal distribution.",
            "hyperparameters": {}
        },
        "MultinomialNB": {
            "type": "Classification",
            "description": "MultinomialNB is a variant of the Naive Bayes algorithm suitable for data with multinomial distributions, such as word counts in text data.",
            "hyperparameters": {
                "alpha": "float (default=1.0)",
                "fit_prior": "bool (default=True)",
            }
        }
    }


@app.route("/")
def home():
    return render_template('index.html', title='Home', message='Hello Users, Welcome!!')


@app.route('/datasets', methods=['GET'])
def get_datasets():
    datasets = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
    return jsonify(datasets)

@app.route('/datasets', methods=['POST'])
def post_dataset():
    dataset_name = request.form['dataset_name']
    dataset_path = os.path.join(DATASETS_DIR, dataset_name)
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        first_10_rows = df.head(10).to_json(orient='records')
        return jsonify({'data': first_10_rows})
    else:
        return jsonify({'error': 'Dataset not found'}), 404

def load_feature_columns():
    if os.path.exists(FEATURE_COLUMNS_FILE):
        with open(FEATURE_COLUMNS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_feature_columns(columns):
        with open(FEATURE_COLUMNS_FILE, 'w') as file:
            json.dump(columns, file)

@app.route('/features', methods=['POST'])
def save_columns():
    data = request.json
    selected_columns = data.get('selected', [])
    # {"selected":["age"]}
    
    print("Selected columns:", selected_columns)

    save_feature_columns(selected_columns)
    
    return jsonify({"success": True, "features": selected_columns})

@app.route('/features', methods=['GET'])
def get_features():
    feature_columns = load_feature_columns()
    return jsonify({"features": feature_columns})

@app.route('/get_unselected_columns', methods=['POST'])
def get_unselected_columns():
    dataset_name = request.form['dataset_name']
    dataset_path = os.path.join(DATASETS_DIR, dataset_name)
    
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        all_columns = set(df.columns)
        selected_columns = set(load_feature_columns())
        unselected_columns = list(all_columns - selected_columns)
        print("Unselected columns:", unselected_columns)
        return jsonify({'unselected_columns': unselected_columns})
    else:
        return jsonify({'error': 'Dataset not found'}), 404

def save_selected_model(model_name):
    with open(SELECTED_MODEL_FILE, 'w') as f:
        f.write(json.dumps({'models':list(model_name)}))
        
def load_selected_model():
    if os.path.exists(SELECTED_MODEL_FILE):
        with open(SELECTED_MODEL_FILE, 'r') as f:
            return json.load(f).get('model', '')
    return ''

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({'models': list(models.keys())})

@app.route('/models', methods=['POST'])
def select_model():
    data = request.json
    selected_model = data.get('model')
    save_selected_model(selected_model)
    print("The selected model:", selected_model)
    return jsonify({'success': True, 'selected_model': selected_model})

@app.route('/model_details', methods=['POST'])
def get_model_details():
    model_name = request.json.get('model')
    print(model_name)
    if model_name:
        if model_name[0] in models:
            return jsonify({'model':models[model_name[0]]})
        else:
            return jsonify({"error": "Model not found"}), 404
    else:
        return jsonify({"error": "Model name not provided"}), 400
    
@app.route("/training", methods=['POST'])
def train():
    model = load_selected_model()
    print(model)
    if model:
        if model[0] == 'LogisticRegression':
            training_model = LogisticRegression()
        elif model[0] == 'MLPClassifier':
            training_model = MLPClassifier()
        elif model[0] == 'SVC':
            training_model = SVC()
    else:
        return jsonify({"error": "Model name not provided"}), 400
    
    if os.path.exists('datasets/heart.csv'):
        df = pd.read_csv('datasets/heart.csv')
        all_columns = set(df.columns)
        selected_columns = set(load_feature_columns())
        print(all_columns, selected_columns)
        unselected_columns = list(all_columns - selected_columns)
        
    x_train = df.drop(columns = unselected_columns)
    y_train = df.drop(columns = selected_columns)
    
    x_train = pd.get_dummies(x_train)
    y_train = pd.get_dummies(y_train)
    
    training_model = training_model.fit(x_train, y_train)
    
    import pickle
    with open(str(model[0])+'.pkl', 'wb') as f:
        pickle.dump(training_model, f)
    
    return jsonify({'success': True})

@app.route("/testing", methods=['POST'])
def test():
    import pickle
    with open('LogisticRegression.pkl', 'rb') as f:
        testing_model = pickle.load(f)

    if os.path.exists('datasets/heart.csv'):
        df = pd.read_csv('datasets/heart.csv')
        all_columns = set(df.columns)
        selected_columns = set(load_feature_columns())
        print(all_columns, selected_columns)
        unselected_columns = list(all_columns - selected_columns)
        
    x_train = df.drop(columns = unselected_columns)
    y_train = df.drop(columns = selected_columns)
    
    x_train = pd.get_dummies(x_train)
    y_train = pd.get_dummies(y_train)
    
    predicted = testing_model.predict(x_train)
    results = confusion_matrix(y_train, predicted) 

    print ('Confusion Matrix :')
    print(results) 
    print ('Accuracy Score :',accuracy_score(y_train, predicted) )
    print ('Report : ')
    print( classification_report(y_train, predicted) )
    return jsonify({'success': True})
