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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertForMaskedLM
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import uuid
import joblib
import matplotlib.pyplot as plt
import io

DATASETS_DIR = 'datasets'
FEATURE_COLUMNS_FILE = 'features.json'
SELECTED_MODEL_FILE = 'selected_model.json'
MC_MODEL = 'GPT2'
HISTOGRAMS_DIR = 'histograms'
dataset_name = None
train_percentage = None
test_percentage = None
selected_model = None
unselected_columns = None

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
    global dataset_name
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
    global unselected_columns
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
            return json.load(f).get('models', '')
    return ''

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({'models': list(models.keys())})

@app.route('/models', methods=['POST'])
def select_model():
    global selected_model
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

#Post the percentage of split data
@app.route('/split_data',methods=['POST'])
def split_data():
    global dataset_name
    if dataset_name is None:
        return jsonify({'error': 'Dataset name not provided'}), 400

    dataset_path = os.path.join(DATASETS_DIR, dataset_name)
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        return jsonify({'error': 'Dataset not found'}), 400

    data = request.json
    train_percentage = data.get('trainPercentage', 0)
    test_percentage = data.get('testPercentage', 0)

    train, test = train_test_split(df, test_size=test_percentage, random_state=4)

    response_data = {
        'message': 'Data split successfully',
        'trainPercentage': train_percentage,
        'testPercentage': test_percentage
    }

    return jsonify({'train_size': len(train), 'test_size': len(test)}), 200

@app.route('/train_test_model', methods=['POST'])
def train_model():
        global dataset_name, unselected_columns, train_percentage, test_percentage, X_test, y_test, model

        if dataset_name is None:
            return jsonify({'error':'Dataset name not provide'}), 400
        
        dataset_path = os.path.join(DATASETS_DIR, dataset_name)
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
        else:
            return jsonify({'error': 'Dataset not found'}), 400
        
        feature_columns = load_feature_columns()
        if not feature_columns:
            return jsonify({'error': 'No feature columns selected'}), 400
        
        if not unselected_columns:
            return jsonify({'error': 'Unselected columns not found'}), 400
        
        target_column = unselected_columns[0]
        if target_column not in df.columns:
            return jsonify({'error': 'Target column not found in dataset'}), 400
        
        X = df[feature_columns]
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage, random_state=4)

        model_name = load_selected_model()
        print(model_name)
        if not model_name:
            return jsonify({'error': 'No model selected'}), 400
        
        model = None
        if model_name[0] == 'LogisticRegression':
            model = LogisticRegression(**models[model_name[0]]['hyperparameters'])
        elif model_name[0] == 'SVC':
            model = SVC(**models[model_name[0]]['hyperparameters'])
        elif model_name[0] == 'MLPClassifier':
            model = MLPClassifier(**models[model_name[0]]['hyperparameters'])
        elif model_name[0] == 'GaussianNB':
            model = GaussianNB()
        elif model_name[0] == 'MultinomialNB':
            model = MultinomialNB(**models[model_name[0]]['hyperparameters'])
        else:
            return jsonify({'error': 'Selected model is not supported'}), 400

        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)

        print(X_train, X_test)
        model.fit(X_train, y_train)

        predicted = model.predict(X_test)
        results = confusion_matrix(y_test, predicted) 

        print ('Confusion Matrix :')
        print(results) 
        score = accuracy_score(y_test, predicted)
        print ('Accuracy Score :', score)
        print ('Report : ')
        print( classification_report(y_test, predicted) )
        # return jsonify({'success': True})

        model_filename = f"trained_model_{uuid.uuid4().hex}.pkl"
        joblib.dump(model, os.path.join(DATASETS_DIR, model_filename))

        # train_data_path = os.path.join(DATASETS_DIR, f"train_data_{uuid.uuid4().hex}.csv")
        # test_data_path = os.path.join(DATASETS_DIR, f"test_data_{uuid.uuid4().hex}.csv")
        # pd.DataFrame(X_train).assign(target=y_train).to_csv(train_data_path, index=False)
        # pd.DataFrame(X_test).assign(target=y_test).to_csv(test_data_path, index=False)

        response_data = {
            'message': 'Model trained successfully',
            'model': model_name,
            'accuracy': score,
            'model_filename': model_filename,
            # 'train_path_data': train_data_path,
            # 'test_path_data': test_data_path
        }

        return jsonify(response_data), 200

#generate an histogram
@app.route('/generate_histogram', methods=['POST'])
def generate_histogram():
    global model, X_test, y_test

    if model is None or X_test is None or y_test is None:
            return jsonify({'error': 'Model or test data not found'}), 400

    plt.hist(y_test)
    # plt.hist(y_test, bins=10, alpha=0.5, label='Test')
    plt.hist(model.predict(X_test), bins = 10, alpha = 0.5, label = 'Predicted')
    plt.legend(loc='upper right')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title('Histogram of actual vs Predicted Classes')
    histogram_path = os.path.join(HISTOGRAMS_DIR, f"histogram_{uuid.uuid4().hex}.png")
    plt.savefig(histogram_path)

    # img = io.BytesIO()
    # plt.savefig(img, format='png')
    # img.seek(0)
    plt.close()

    return jsonify('success', True)

def gpt2_generate_response(input_text):
        # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output = model.generate(input_ids, max_length=100, num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text

def bert_generate_text(prompt):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


@app.route("/mc_models", methods=['POST', 'GET'])
def chat_functionality():
    global MC_MODEL
    MC_MODEL = 'BERT-BASE-UCASED'
    if request.method == 'POST':
        user_input = request.json.get('user_input', '')
        if MC_MODEL:
            response = gpt2_generate_response(user_input)
        else:
            response = bert_generate_text(user_input)
        return jsonify({"response": response})
