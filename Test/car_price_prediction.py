import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBRegressor,XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('car_sales_data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}\n")

# Drop missing values
df = df.dropna()

# Identify feature types
categorical_cols = ['Manufacturer', 'Model', 'Fuel type']
numeric_cols = ['Engine size', 'Year of manufacture', 'Mileage']
target = 'Price'

# Add engineered features
df['Age'] = 2025 - df['Year of manufacture']
df['Mileage_per_year'] = df['Mileage'] / (df['Age'] + 1)
df['Engine_power_estimate'] = df['Engine size'] * 50
numeric_cols_extended = numeric_cols + ['Age', 'Mileage_per_year', 'Engine_power_estimate']

# Features and target for regression
X = df[categorical_cols + numeric_cols_extended]
y = df[target]

# Create price categories for classification (3 categories: cheap, medium, expensive)
price_categories_labels = pd.qcut(y, q=3, labels=['Cheap', 'Medium', 'Expensive'])
price_categories = pd.qcut(y, q=3, labels=[0, 1, 2])  # Convert to numeric

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_, _, y_train_cat, y_test_cat = train_test_split(X, price_categories, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Price categories distribution:\n{price_categories_labels.value_counts()}\n")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numeric_cols_extended)
    ]
)

regression_results = {}
classification_results = {}

# ============ REGRESSION MODELS ============
print("=" * 90)
print("REGRESSION MODELS - Predicting Price")
print("=" * 90)

# 1. Linear Regression
print("Training Linear Regression...")
lr = Pipeline(steps=[('preprocessor', preprocessor), ('model', LinearRegression())])
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
regression_results['Linear Regression'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
    'MAE': mean_absolute_error(y_test, lr_pred),
    'R2': r2_score(y_test, lr_pred)
}

# 2. Decision Tree Regressor
print("Training Decision Tree...")
dt_reg = Pipeline(steps=[('preprocessor', preprocessor), ('model', DecisionTreeRegressor(max_depth=10, random_state=42))])
dt_reg.fit(X_train, y_train)
dt_reg_pred = dt_reg.predict(X_test)
regression_results['Decision Tree'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, dt_reg_pred)),
    'MAE': mean_absolute_error(y_test, dt_reg_pred),
    'R2': r2_score(y_test, dt_reg_pred)
}

# 3. K-Nearest Neighbors
print("Training KNN...")
knn_reg = Pipeline(steps=[('preprocessor', preprocessor), ('model', KNeighborsRegressor(n_neighbors=5))])
knn_reg.fit(X_train, y_train)
knn_reg_pred = knn_reg.predict(X_test)
regression_results['KNN'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, knn_reg_pred)),
    'MAE': mean_absolute_error(y_test, knn_reg_pred),
    'R2': r2_score(y_test, knn_reg_pred)
}

# 4. Neural Network
print("Training Neural Network...")
mlp_reg = Pipeline(steps=[('preprocessor', preprocessor), ('model', MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True))])
mlp_reg.fit(X_train, y_train)
mlp_reg_pred = mlp_reg.predict(X_test)
regression_results['Neural Network'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, mlp_reg_pred)),
    'MAE': mean_absolute_error(y_test, mlp_reg_pred),
    'R2': r2_score(y_test, mlp_reg_pred)
}

# 5. Gradient Boosting Regressor
print("Training Gradient Boosting...")
gb_reg = Pipeline(steps=[('preprocessor', preprocessor), ('model', GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42))])
gb_reg.fit(X_train, y_train)
gb_reg_pred = gb_reg.predict(X_test)
regression_results['Gradient Boosting'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, gb_reg_pred)),
    'MAE': mean_absolute_error(y_test, gb_reg_pred),
    'R2': r2_score(y_test, gb_reg_pred)
}

# 6. Support Vector Machine (SVM)
print("Training SVM Regressor...")
svr = Pipeline(steps=[('preprocessor', preprocessor), ('model', SVR(kernel='rbf', C=1000, gamma=0.1))])
svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)
regression_results['SVM'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, svr_pred)),
    'MAE': mean_absolute_error(y_test, svr_pred),
    'R2': r2_score(y_test, svr_pred)
}

# 7. XGBoost
print("Training XGBoost...")
xgb_reg = Pipeline(steps=[('preprocessor', preprocessor), ('model', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4))])
xgb_reg.fit(X_train, y_train)
xgb_reg_pred = xgb_reg.predict(X_test)
regression_results['XGBoost'] = {
    'RMSE': np.sqrt(mean_squared_error(y_test, xgb_reg_pred)),
    'MAE': mean_absolute_error(y_test, xgb_reg_pred),
    'R2': r2_score(y_test, xgb_reg_pred)
}


# ============ CLASSIFICATION MODELS ============
print("\n" + "=" * 90)
print("CLASSIFICATION MODELS - Predicting Price Category (Cheap/Medium/Expensive)")
print("=" * 90)

# 1. Logistic Regression
print("Training Logistic Regression...")
log_reg = Pipeline(steps=[('preprocessor', preprocessor), ('model', LogisticRegression(max_iter=1000, random_state=42))])
log_reg.fit(X_train, y_train_cat)
log_reg_pred = log_reg.predict(X_test)
classification_results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test_cat, log_reg_pred),
    'Precision': precision_score(y_test_cat, log_reg_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test_cat, log_reg_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test_cat, log_reg_pred, average='weighted', zero_division=0)
}

# 2. Naive Bayes
print("Training Naive Bayes...")
nb = Pipeline(steps=[('preprocessor', preprocessor), ('model', GaussianNB())])
nb.fit(X_train, y_train_cat)
nb_pred = nb.predict(X_test)
classification_results['Naive Bayes'] = {
    'Accuracy': accuracy_score(y_test_cat, nb_pred),
    'Precision': precision_score(y_test_cat, nb_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test_cat, nb_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test_cat, nb_pred, average='weighted', zero_division=0)
}

# 3. Decision Tree Classifier
print("Training Decision Tree Classifier...")
dt_cls = Pipeline(steps=[('preprocessor', preprocessor), ('model', DecisionTreeClassifier(max_depth=10, random_state=42))])
dt_cls.fit(X_train, y_train_cat)
dt_cls_pred = dt_cls.predict(X_test)
classification_results['Decision Tree'] = {
    'Accuracy': accuracy_score(y_test_cat, dt_cls_pred),
    'Precision': precision_score(y_test_cat, dt_cls_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test_cat, dt_cls_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test_cat, dt_cls_pred, average='weighted', zero_division=0)
}

# 4. Neural Network Classifier
print("Training Neural Network Classifier...")
mlp_cls = Pipeline(steps=[('preprocessor', preprocessor), ('model', MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True))])
mlp_cls.fit(X_train, y_train_cat)
mlp_cls_pred = mlp_cls.predict(X_test)
classification_results['Neural Network'] = {
    'Accuracy': accuracy_score(y_test_cat, mlp_cls_pred),
    'Precision': precision_score(y_test_cat, mlp_cls_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test_cat, mlp_cls_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test_cat, mlp_cls_pred, average='weighted', zero_division=0)
}

# 5. Gradient Boosting Classifier
print("Training Gradient Boosting Classifier...")
gb_cls = Pipeline(steps=[('preprocessor', preprocessor), ('model', GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42))])
gb_cls.fit(X_train, y_train_cat)
gb_cls_pred = gb_cls.predict(X_test)
classification_results['Gradient Boosting'] = {
    'Accuracy': accuracy_score(y_test_cat, gb_cls_pred),
    'Precision': precision_score(y_test_cat, gb_cls_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test_cat, gb_cls_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test_cat, gb_cls_pred, average='weighted', zero_division=0)
}

# 6. Support Vector Machine Classifier (SVM)
print("Training SVM Classifier...")
svc = Pipeline(steps=[('preprocessor', preprocessor), ('model', SVC(kernel='rbf', C=1000, gamma=0.1, random_state=42))])
svc.fit(X_train, y_train_cat)
svc_pred = svc.predict(X_test)
classification_results['SVM'] = {
    'Accuracy': accuracy_score(y_test_cat, svc_pred),
    'Precision': precision_score(y_test_cat, svc_pred, average='weighted', zero_division=0),
    'Recall': recall_score(y_test_cat, svc_pred, average='weighted', zero_division=0),
    'F1': f1_score(y_test_cat, svc_pred, average='weighted', zero_division=0)
}



# ============ PRINT RESULTS ============
print("\n\n" + "=" * 90)
print("REGRESSION RESULTS - Price Prediction")
print("=" * 90 + "\n")

print(f"{'Model':<25} {'RMSE':>15} {'MAE':>15} {'R2':>15}")
print("-" * 90)
sorted_reg = sorted(regression_results.items(), key=lambda x: x[1]['R2'], reverse=True)
for model_name, metrics in sorted_reg:
    print(f"{model_name:<25} ${metrics['RMSE']:>14.2f} ${metrics['MAE']:>14.2f} {metrics['R2']:>15.4f}")
print("-" * 90)
best_reg = sorted_reg[0]
print(f"\n🏆 BEST REGRESSION MODEL: {best_reg[0]} (R2 = {best_reg[1]['R2']:.4f})\n")

print("\n" + "=" * 90)
print("CLASSIFICATION RESULTS - Price Category Prediction")
print("=" * 90 + "\n")

print(f"{'Model':<25} {'Accuracy':>15} {'Precision':>15} {'Recall':>15} {'F1':>15}")
print("-" * 90)
sorted_cls = sorted(classification_results.items(), key=lambda x: x[1]['F1'], reverse=True)
for model_name, metrics in sorted_cls:
    print(f"{model_name:<25} {metrics['Accuracy']:>15.4f} {metrics['Precision']:>15.4f} {metrics['Recall']:>15.4f} {metrics['F1']:>15.4f}")
print("-" * 90)
best_cls = sorted_cls[0]
print(f"\n🏆 BEST CLASSIFICATION MODEL: {best_cls[0]} (F1 = {best_cls[1]['F1']:.4f})\n")

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"\nBest Regression Model: {best_reg[0]}")
print(f"  - RMSE: ${best_reg[1]['RMSE']:.2f} (average prediction error)")
print(f"  - MAE:  ${best_reg[1]['MAE']:.2f} (typical error)")
print(f"  - R2:   {best_reg[1]['R2']:.4f} (variance explained)")

print(f"\nBest Classification Model: {best_cls[0]}")
print(f"  - Accuracy:  {best_cls[1]['Accuracy']:.4f} (correct predictions)")
print(f"  - Precision: {best_cls[1]['Precision']:.4f} (false positive rate)")
print(f"  - Recall:    {best_cls[1]['Recall']:.4f} (true positive rate)")
print(f"  - F1:        {best_cls[1]['F1']:.4f} (balance of precision/recall)")

# ============ FEATURE IMPORTANCE ANALYSIS ============
print("\n" + "=" * 90)
print("FEATURE IMPORTANCE ANALYSIS - Base Features Only")
print("=" * 90)

# Define base features (original dataset features only)
base_features = ['Manufacturer', 'Model', 'Engine size', 'Fuel type', 'Year of manufacture', 'Mileage']

# For XGBoost Regression - Aggregate Feature Importance by Base Feature
if 'XGBoost' in regression_results:
    print("\nAnalyzing XGBoost Regression feature importance...")
    
    # Get feature names after preprocessing
    xgb_model = xgb_reg.named_steps['model']
    
    # Get feature names from preprocessor
    cat_features = xgb_reg.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    num_features = numeric_cols_extended
    all_features = list(cat_features) + num_features
    
    # Get feature importances
    importances = xgb_model.feature_importances_
    
    # Aggregate by base feature
    base_feature_importance = {}
    for feature_name, importance in zip(all_features, importances):
        # Map to base feature
        if feature_name in ['Engine size', 'Year of manufacture', 'Mileage']:
            base_feature = feature_name
        elif feature_name in ['Age', 'Mileage_per_year', 'Engine_power_estimate']:
            # Skip engineered features
            continue
        else:
            # For categorical (e.g., "Manufacturer_BMW" -> "Manufacturer")
            base_feature = feature_name.split('_')[0]
            if base_feature == 'Fuel':
                base_feature = 'Fuel type'
        
        if base_feature in base_features:
            if base_feature not in base_feature_importance:
                base_feature_importance[base_feature] = 0
            base_feature_importance[base_feature] += importance
    
    # Convert to DataFrame and sort
    xgb_base_df = pd.DataFrame(list(base_feature_importance.items()), 
                                columns=['Feature', 'Importance'])
    xgb_base_df = xgb_base_df.sort_values('Importance', ascending=False)
    xgb_base_df['Percentage'] = (xgb_base_df['Importance'] / xgb_base_df['Importance'].sum()) * 100
    
    print("\nXGBoost Regression - Base Feature Importance:")
    print("-" * 90)
    print(f"{'Feature':<30} {'Importance':>20} {'Percentage':>20}")
    print("-" * 90)
    for _, row in xgb_base_df.iterrows():
        print(f"{row['Feature']:<30} {row['Importance']:>20.4f} {row['Percentage']:>19.2f}%")
    
    # Plot XGBoost Base Feature Importance
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(xgb_base_df)))
    bars = plt.barh(range(len(xgb_base_df)), xgb_base_df['Importance'], color=colors)
    plt.yticks(range(len(xgb_base_df)), xgb_base_df['Feature'], fontsize=12)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title('XGBoost Regression - Base Feature Importance\n(Impact on Price)', 
              fontweight='bold', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, xgb_base_df['Percentage'])):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)

# For SVM Classification - Aggregate Permutation Importance by Base Feature
if 'SVM' in classification_results:
    print("\n\nAnalyzing SVM Classification feature importance...")
    
    from sklearn.inspection import permutation_importance
    
    # Calculate permutation importance for SVM
    X_test_transformed = svc.named_steps['preprocessor'].transform(X_test)
    perm_importance = permutation_importance(svc.named_steps['model'], X_test_transformed, y_test_cat, 
                                             n_repeats=10, random_state=42, n_jobs=4)
    
    # Get feature names
    cat_features = svc.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    num_features = numeric_cols_extended
    all_features_svm = list(cat_features) + num_features
    
    # Aggregate by base feature
    svm_base_feature_importance = {}
    for feature_name, importance in zip(all_features_svm, perm_importance.importances_mean):
        # Map to base feature
        if feature_name in ['Engine size', 'Year of manufacture', 'Mileage']:
            base_feature = feature_name
        elif feature_name in ['Age', 'Mileage_per_year', 'Engine_power_estimate']:
            # Skip engineered features
            continue
        else:
            # For categorical (e.g., "Manufacturer_BMW" -> "Manufacturer")
            base_feature = feature_name.split('_')[0]
            if base_feature == 'Fuel':
                base_feature = 'Fuel type'
        
        if base_feature in base_features:
            if base_feature not in svm_base_feature_importance:
                svm_base_feature_importance[base_feature] = 0
            svm_base_feature_importance[base_feature] += importance
    
    # Convert to DataFrame and sort
    svm_base_df = pd.DataFrame(list(svm_base_feature_importance.items()), 
                                columns=['Feature', 'Importance'])
    svm_base_df = svm_base_df.sort_values('Importance', ascending=False)
    svm_base_df['Percentage'] = (svm_base_df['Importance'] / svm_base_df['Importance'].sum()) * 100
    
    print("\nSVM Classification - Base Feature Importance:")
    print("-" * 90)
    print(f"{'Feature':<30} {'Importance':>20} {'Percentage':>20}")
    print("-" * 90)
    for _, row in svm_base_df.iterrows():
        print(f"{row['Feature']:<30} {row['Importance']:>20.4f} {row['Percentage']:>19.2f}%")
    
    # Plot SVM Base Feature Importance
    plt.subplot(2, 1, 2)
    colors = plt.cm.plasma(np.linspace(0, 1, len(svm_base_df)))
    bars = plt.barh(range(len(svm_base_df)), svm_base_df['Importance'], color=colors)
    plt.yticks(range(len(svm_base_df)), svm_base_df['Feature'], fontsize=12)
    plt.xlabel('Importance Score (Permutation)', fontsize=12, fontweight='bold')
    plt.title('SVM Classification - Base Feature Importance\n(Impact on Price Category)', 
              fontweight='bold', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, svm_base_df['Percentage'])):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{pct:.1f}%', va='center', fontsize=10, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_importance_base_features.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: feature_importance_base_features.png")
    plt.show()

print("\n" + "=" * 90)
print("Analysis complete! Check 'feature_importance_base_features.png' for visual results.")
print("=" * 90)

