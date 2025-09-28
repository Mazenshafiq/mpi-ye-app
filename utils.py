
# -*- coding: utf-8 -*-
"""
دوال مساعدة لتحميل النموذج والتعامل مع الملفات والاستدلال على مخطط الميزات.
"""
import io
import pickle
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

try:
    import joblib
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

def load_model(file_bytes: bytes):
    """تحميل نموذج سكيت-ليرن (Pipeline/Estimator) من ملف .pkl أو .joblib."""
    buf = io.BytesIO(file_bytes)
    # نجرب joblib أولاً إن توفر
    if _HAS_JOBLIB:
        try:
            buf.seek(0)
            return joblib.load(buf)
        except Exception:
            pass
    # نرجع إلى pickle
    buf.seek(0)
    return pickle.load(buf)

def read_table(uploaded_file) -> pd.DataFrame:
    """قراءة ملف Excel/CSV إلى DataFrame مع محاولة اكتشاف الترميز تلقائياً."""
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    # CSV / TSV
    uploaded_file.seek(0)
    # حاول كشف الفاصل
    data = uploaded_file.read()
    # نعيد المؤشر
    buf = io.BytesIO(data)
    try:
        df = pd.read_csv(buf)
        return df
    except Exception:
        buf.seek(0)
        try:
            df = pd.read_csv(buf, sep=";")
            return df
        except Exception:
            buf.seek(0)
            df = pd.read_csv(buf, sep="\t")
            return df

def is_classifier(model) -> bool:
    """محاولة تحديد إن كان النموذج تصنيفياً."""
    # سكيت-ليرن يعرّف خاصية _estimator_type
    est_type = getattr(model, "_estimator_type", None)
    if est_type == "classifier":
        return True
    if est_type == "regressor":
        return False
    # إن لم تتوفر، نتحقق من وجود predict_proba
    return hasattr(model, "predict_proba")

def feature_names_from_model(model) -> Optional[List[str]]:
    """إرجاع أسماء الميزات المتوقعة من النموذج إن توفرت."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    # حاول الوصول إلى ColumnTransformer داخل Pipeline
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    if isinstance(model, Pipeline):
        for step_name, step in model.named_steps.items():
            if isinstance(step, ColumnTransformer):
                cols = []
                for trans in step.transformers_:
                    # trans: (name, transformer, columns)
                    if len(trans) >= 3:
                        cols_part = trans[2]
                        if isinstance(cols_part, (list, tuple, np.ndarray, pd.Index)):
                            cols.extend(list(cols_part))
                if cols:
                    return list(dict.fromkeys(cols))
    return None

def infer_schema_from_sample(sample_df: pd.DataFrame, expected_cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    إنشاء مخطط (schema) مبسط للميزات من DataFrame عينة.
    لكل عمود: name, type ('numeric'/'integer'/'category'/'text'), choices (اختياري).
    """
    schema = []
    cols = list(sample_df.columns)
    if expected_cols:
        # نحافظ على ترتيب المتوقع، مع أي أعمدة إضافية في النهاية
        cols = [c for c in expected_cols if c in sample_df.columns] + [c for c in cols if c not in (expected_cols or [])]

    for col in cols:
        ser = sample_df[col]
        if pd.api.types.is_integer_dtype(ser):
            col_type = "integer"
            schema.append({"name": col, "type": col_type})
        elif pd.api.types.is_numeric_dtype(ser):
            col_type = "numeric"
            schema.append({"name": col, "type": col_type})
        elif ser.nunique(dropna=True) <= 20:
            # أصناف قليلة => اختيار من قائمة
            choices = sorted([str(x) for x in ser.dropna().unique().tolist()])
            schema.append({"name": col, "type": "category", "choices": choices})
        else:
            schema.append({"name": col, "type": "text"})
    return schema

def cast_inputs(df: pd.DataFrame, schema: List[Dict[str, Any]]) -> pd.DataFrame:
    """تحويل أنواع بيانات الإدخال حسب الـ schema قدر الإمكان."""
    df = df.copy()
    for item in schema:
        name = item["name"]
        typ = item.get("type", "text")
        if name not in df.columns:
            continue
        if typ in ("numeric", "integer"):
            df[name] = pd.to_numeric(df[name], errors="coerce")
            if typ == "integer":
                df[name] = df[name].round().astype("Int64")
        elif typ == "category":
            # اتركها كنص؛ غالباً سيعالجها Pipeline (OneHotEncoder) إن وُجد
            df[name] = df[name].astype("string")
        else:
            df[name] = df[name].astype("string")
    return df

def predict_with_model(model, X: pd.DataFrame):
    """إجراء التنبؤ، وإرجاع dict يحتوي على y_pred، y_proba (إن توفرت) و class_names (إن وُجدت)."""
    out = {}
    # الترتيب وفقاً لِ feature_names_in_ إن توفرت
    expected = feature_names_from_model(model)
    if expected:
        # بعض النماذج قد تتجاهل الأعمدة الإضافية، لكن الأفضل المطابقة
        missing = [c for c in expected if c not in X.columns]
        if missing:
            raise ValueError(f"أعمدة مفقودة في البيانات: {missing}")
        X = X.reindex(columns=expected)
    # التنبؤ
    y_pred = model.predict(X)
    out["y_pred"] = y_pred
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            out["y_proba"] = proba
            classes = getattr(model, "classes_", None)
            out["class_names"] = classes
        except Exception:
            pass
    return out
