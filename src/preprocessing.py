from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_tabular_data(x_train):
    num_features = x_train.select_dtypes(include="number").columns.tolist()
    cat_features = [column for column in x_train.columns if column not in num_features]

    transformers = []
    if num_features:
        transformers.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_features))
    if cat_features:
        transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_features))

    if not transformers:
        raise ValueError("x_train must contain at least one feature")
    preprocessor = ColumnTransformer(transformers=transformers)

    x_train_processed = preprocessor.fit_transform(x_train)
    return preprocessor, x_train_processed
