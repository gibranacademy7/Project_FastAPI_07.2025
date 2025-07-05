import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def train_model_from_file(csv_filename: str):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, csv_filename)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # الأعمدة المطلوبة حسب الملف
    df = df[['Age', 'Music Instrument', 'Lesson Price', 'Annual Income (Nis.)']].dropna()
    df.columns = ['age', 'instrument', 'lesson_price', 'income']

    df_encoded = pd.get_dummies(df, columns=['instrument', 'lesson_price'])

    X = df_encoded.drop('income', axis=1)
    y = df_encoded['income']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    model = LinearRegression()
    model.fit(X_train, y_train)

    model_path = os.path.join(base_dir, 'saved_models', 'model.pkl')
    joblib.dump((model, X.columns.tolist()), model_path)

    return {
        'mse': mean_squared_error(y_test, model.predict(X_test)),
        'r2': r2_score(y_test, model.predict(X_test))
    }

def predict_income(age: float, instrument: str, lesson_price: int):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'saved_models', 'model.pkl')
    model, feature_names = joblib.load(model_path)

    data = {'age': age, 'instrument': instrument, 'lesson_price': lesson_price}
    df = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df)

    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[feature_names]

    return model.predict(df_encoded)[0]
