import pandas as pd
import lightgbm as lgb
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import os

# Asegúrate de que LightGBM está instalado: pip install lightgbm

def train_and_evaluate_champion():
    # 1. CARGA DE DATOS PROCESADOS
    try:
        # Asegúrate de usar la ruta correcta a tu archivo procesado
        df = pd.read_parquet('../artifacts/train_processed.parquet') 
    except FileNotFoundError:
        print("ERROR: Archivo procesado no encontrado. Ejecuta el script de Data Preparation primero.")
        return

    X = df.drop(columns=['TARGET'])
    y = df['TARGET']

    # --- 2. MANEJO DEL DESBALANCE (CRITERIO CLAVE) ---
    # Calcular la proporción de desbalance (ej. si 8% es la clase positiva)
    # Ratio = (Cantidad de Clase 0) / (Cantidad de Clase 1)
    # LightGBM usará este peso para dar más importancia a la clase minoritaria (incumplimiento).
    class_ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    print(f"Ratio de Desbalance (Clase 0 / Clase 1): {class_ratio:.2f}")

    # --- 3. VALIDACIÓN ROBUSTA (Stratified K-Fold) ---
    # Usar Stratified K-Fold asegura que cada pliegue de entrenamiento y validación
    # mantenga la misma proporción de la variable TARGET (el desbalance).
    N_FOLDS = 5
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # Lista para guardar los modelos entrenados en cada fold
    models = []
    oof_preds = np.zeros(X.shape[0]) # Out-Of-Fold predictions para evaluación

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        # 4. ENTRENAMIENTO DEL MODELO (LightGBM)
        lgb_clf = lgb.LGBMClassifier(
            objective='binary',
            metric='auc', # AUC es la métrica más adecuada para desbalance
            n_estimators=10000,
            learning_rate=0.01,
            # Parámetro CLAVE para el desbalance:
            scale_pos_weight=class_ratio, 
            random_state=42,
            n_jobs=-1,
            verbose=-1 # Silenciar salida para limpieza
        )

        lgb_clf.fit(X_train, y_train, 
                    eval_set=[(X_valid, y_valid)],
                    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=-1)])

        oof_preds[valid_idx] = lgb_clf.predict_proba(X_valid)[:, 1]
        models.append(lgb_clf)
        
    # --- 5. EVALUACIÓN FINAL DEL MODELO (Basado en OOF) ---
    final_auc = roc_auc_score(y, oof_preds)
    print(f"\n--- Resultados de Validación (5 Folds) ---")
    print(f"AUC-ROC Final del Modelo: {final_auc:.4f}")

    # Seleccionamos el modelo con el mejor rendimiento (o el último, para este ejemplo)
    champion_model = models[-1] 

    # 6. GUARDAR EL MODELO CAMPEÓN
    model_path = '../artifacts/final_lgbm_model.pkl'
    joblib.dump(champion_model, model_path)
    print(f"Modelo campeón guardado en {model_path}")
    
    # Guardamos los OOF predictions para el script de evaluación
    pd.Series(oof_preds, name='OOF_PREDS').to_csv('../artifacts/oof_predictions.csv', index=False)
    
    return champion_model

if __name__ == '__main__':
    train_and_evaluate_champion()# Comando para instalar Pandas, LightGBM, Scikit-learn, etc.
