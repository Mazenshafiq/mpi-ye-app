
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import json
import pandas as pd
import streamlit as st
import joblib
import pickle as pkl
import io
import base64
from io import BytesIO
from streamlit.components.v1 import html as st_html

from utils import load_model, read_table, feature_names_from_model, infer_schema_from_sample, cast_inputs, predict_with_model


st.set_page_config(
    page_title="واجهة تنبؤ نموذج التعلم الآلي",
    page_icon="🤖",
    layout="wide"
)

# قائمة الصفحات
pages = st.sidebar.radio("انتقل إلى:", ["حول ","📥 تنبأ بأسرة واحدة", "📊 تنبأ لأكثر من اسرة واحدة"])

# الواجهة الأولى
if pages == "حول ":
   st.title("مرحباً بك 👋")
   st.markdown("""

    هنا صفحة التنبؤ لنموذج الفقر متعدد الابعاد الخاص بالأسر اليمنية تتوفر خدمتنا عبر طريقتين هما:
    1. **إدخال بيانات أسرة واحدة* للحصول على نتيجة فورية.
    2. **تنبؤ لأسر متعددة** عبر رفع ملف Excel/CSV، وسيتم إضافة عمود بالتنبؤات لتنزيله.

    > تلميحات سريعة:
    > - في كل صفحة يمكنك رفع ملف النموذج المدرب ان كان لديك  نموذج مدرب (`.pkl` أو `.joblib`).
    > - يوجد في الاسفل تعريفات عن المتغيرات و القيم المدخلة.

    """)

   instructions = pd.DataFrame({
        "المتغير": ["HH48A_HC3", "FC6A", "HC7A", "HC7B", "HC8", "HC9A", 
                    "HC9B", "HC9C", "HC10B", "HC10C", "HC10E", "HC11", "HC12", "HC13", "HC14", "HC15", "HC17",
                    "EU4", "WS1", "WS4", "WS7", "WS11", "WS15", "NU_malnutrition_hh", "HE_earlychild_hh", "HE_immune_hh", "HE_antecare_hh", "HE_birthassist_hh",
                        "ED_attendance_hh", "ED_completion_hh:", "ED_gradeforage_hh", "ED_adult_hh"],
        "الوصف": [
            "ناتج قسمة عدد الافراد على عدد الغرف",
            "هل حدث هذا خلال الأسبوعين الماضيين (30 يومًا)؟",
            "هل لدى الأسرة خط هاتف أرضي؟",
            "هل لدى الأسرة جهاز راديو؟",
            "هل متوفر لدى الأسرة إمداد كهربائي؟",
            "هل لدى الأسرة جهاز تلفزيون؟",
            "هل لدى الأسرة ثلاجة؟",
            "هل لدى الأسرة غسالة ملابس؟",
            "هل يملك أحد أفراد الأسرة دراجة هوائية؟",
            "هل يملك أحد أفراد الأسرة دراجة نارية أو دراجة بخارية؟",
            "هل يملك أحد أفراد الأسرة سيارة أو شاحنة أو باص صغير؟",
            " هل يملك أحد أفراد الأسرة جهاز كمبيوتر؟",
            "هل يملك أحد أفراد الأسرة هاتفًا محمولًا؟",
            "هل يتوفر لدى الأسرة خدمة إنترنت في المنزل؟",
            "هل تمتلك الأسرة المنزل الذي تسكن فيه؟",
            "هل يملك أحد أفراد الأسرة أرضًا زراعية؟",
            "هل تمتلك الأسرة أي حيوانات؟",
            "نوع مصدر الطاقة المستخدم في الموقد",
            "المصدر الرئيسي لمياه الشرب",
            "الوقت اللازم (بالدقائق) للحصول على المياه والعودة",
            "هل حدث نقص في إمدادات المياه خلال الشهر الماضي؟",
            " نوع المرافق الصحية",
            "هل المرافق الصحية مشتركة؟",
            "هل هناك اطفال يتراوح اعمارهم بين 0 و 59 شهراً يعاني من قصور في النمو او يعاني من نقص في الوزن ",
            " هل هناك إمرأة في الاسرة انجبت طفلاً حياً خلال العامين السابقين عندما كانت دون سن 18 عاماً",
            "هل هناك اطفال يتراوح اعمارهم بين 12 و35 شهراً ولم يتلقوا جرعات لقاح DPT1 و DPT2 و DPT3 ",
            "هل تلقت الأم الرعاية الصحية  اللازمة خلال فترة الحمل (زيارة الطيبيب على الاقل 4 مرات )",
            "هل هناك امرأة في اسرتك لم تلد تحت إراف كوادر صحية مؤهلة",
            "هل هناك طفل يتراوح عمره بين 6 و 12 عاماً لا يذهب الى المدرسة",
            " هل هناك طفل يتراوح عمره بين 14 و 17 عاماً لم يكمل السنوات الست الأولى من التعليم",
            "هل هناك طفل يقل مستواه الدراسي عن المستوى المناسب لعمره بأكثر من عامين",
            "هل هناك احد من افراد الاسرة على الأقل تتجاوز اعمارهم 17 عاماً لم يكملوا السنوات الست الأولى من التعليم "
        ]
    })

   st.subheader("ℹ️ تعريفات المتغيرات")
   st.table(instructions)

    # 🔹 جدول التعليمات في الأعلى
   instructions = pd.DataFrame({
        "المتغير": ["HH48A_HC3", "FC6A", "HC7A", "HC7B", "HC8", "HC9A", 
                    "HC9B", "HC9C", "HC10B", "HC10C", "HC10E", "HC11", "HC12", "HC13", "HC14", "HC15", "HC17",
                    "EU4", "WS1", "WS7", "WS11", "WS15", "NU_malnutrition_hh", "HE_immune_hh", "HE_antecare_hh", "HE_birthassist_hh",
                        "ED_attendance_hh", "ED_completion_hh:", "ED_gradeforage_hh", "ED_adult_hh"],
        "الوصف": [
            "يحسب الطفل الذي يتراوح عمره بين 0 و 4 اعوام نصف فرد وما دون ذلك يعتبر فرد",
            "  لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            "نعم، شبكة خاصة → 3  نعم، خارج الشبكة (مولد/نظام معزول) → 2  نعم، شبكة مترابطة → 1  لا كهرباء → 0 نعم، طاقة شمسية → 4",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا يستخدم وقود صلب → 0  ,يستخدم وقود صلب → 1",
            " مياه بالأنابيب: مياه موصولة إلى المسكن → 5  مياه معبأة في أكياس→ 4 مياه معبأة: مياه معبأة → 3  بئر محفور: بئر غير محمي→ 2  بئر محفور: بئر محمي→ 1  عربة بخزان صغير→ 0   بئر أنبوبي / بئر حفر → 14  شاحنة صهريج → 13  مياه سطحية (نهر، سد، بحيرة، بركة، مجرى مائي، قناة، قناة ري)→ 12  نبع غير محمي → 11  نبع محمي → 10  مياه الأمطار →9   مياه بالأنابيب: صنبور عام/أنبوب عمودي  → 8  مياه بالأنابيب: مياه إلى الفناء/قطعة الأرض → 7  مياه بالأنابيب: مياه إلى الجار → 6",
            " لا، دائمًا كافٍ → 0  نعم، مرة واحدة على الأقل→ 1",
            " مرحاض الحفرة: مرحاض الحفرة المُحسّن المُهوى → 8 مرحاض الحفرة: مرحاض الحفرة بدون بلاطة / حفرة مفتوحة → 7  مرحاض الحفرة: مرحاض الحفرة مع بلاطة → 6  لا يوجد مرفق / شجيرة / حقل→ 5  شطف/صب: شطف إلى مرحاض الحفرة → 4  شطف/صب: شطف إلى نظام الصرف الصحي → 3  شطف/صب: شطف إلى مصرف مفتوح → 2  شطف/صب: شطف إلى حيث لا اعرف → 1  دلو → 0",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1",
            " لا → 0  نعم → 1"
        ]
    })

   st.subheader("ℹ️ تعليمات الإدخال")
   st.table(instructions)

   st.info("من الشريط الجانبي لِكل صفحة: ارفع النموذج وعينة للبيانات إن رغبت. ثم اتبع الخطوات الظاهرة.")
   st.markdown("---")
   st.markdown("ابدأ من تبويب الصفحات في الشريط الجانبي 👈")
#_____________________________________________________________________________________________
# الواجهة الثانية
elif pages == "📥 تنبأ بأسرة واحدة":

    st.set_page_config(page_title="تنبؤ صف مفرد", page_icon="🔢", layout="wide")

    st.title("🔢 تنبؤ أسرة واحدة")
    st.caption("ادخل بيانات أسرة واحدة واحصل على التنبؤ فوراً")

    # --------------------------
    # إعداد النموذج
    # --------------------------
    MODEL_PATH = "model/gboosting_model 25-9-2025.joblib"

    @st.cache_resource
    def load_model():
        try:
            obj = joblib.load(MODEL_PATH)
            # إذا حفظت قاموس يحتوي pipeline
            if isinstance(obj, dict) and "pipeline" in obj:
                return obj["pipeline"]
            return obj
        except Exception as e:
            st.error(f"تعذر تحميل النموذج: {e}")
            return None

    model = load_model()
    if model is not None:
        st.success("تم تحميل النموذج بنجاح ✅")
    else:
        st.stop()  # توقف إذا النموذج لم يُحمّل

    # --------------------------
    # تجهيز المخطط (schema)
    # --------------------------
    expected_cols = feature_names_from_model(model)
    schema = [{"name": c, "type": "numeric"} for c in expected_cols]  # افتراض جميع الأعمدة رقمية

    st.subheader("🧾 أدخل القيم")
    cols = st.columns(6)
    values = {}

    for i, item in enumerate(schema):
        name = item["name"]
        typ = item.get("type", "text")
        choices = item.get("choices", None)
        with cols[i % 6]:
            if typ == "integer":
                values[name] = st.number_input(f"{name}", value=0, step=1)
            elif typ == "numeric":
                values[name] = st.number_input(f"{name}", min_value=0, step=1)
            elif typ == "category" and choices:
                values[name] = st.selectbox(f"{name}", choices=choices)
            else:
                values[name] = st.text_input(f"{name}", value="")

    # تحويل القيم إلى DataFrame
    row_df = pd.DataFrame([values])
    row_df = cast_inputs(row_df, schema)

    # --------------------------
    # زر التنبؤ
    # --------------------------
    if st.button("تنفيذ التنبؤ الآن ✅", use_container_width=True):
        try:
            out = predict_with_model(model, row_df)
            y_pred = out.get("y_pred")
            st.success(f"الناتج: **{y_pred[0]}**")

            if "y_proba" in out:
                proba = out["y_proba"]
                class_names = out.get("class_names")
                proba_df = pd.DataFrame(
                    proba,
                    columns=[str(c) for c in (class_names if class_names is not None else range(proba.shape[1]))]
                )
                st.dataframe(proba_df.style.format("{:.3f}"))

            with st.expander("عرض بيانات الإدخال كما استقبلها النموذج"):
                st.dataframe(row_df)

        except Exception as e:
            st.error(f"حصل خطأ أثناء التنبؤ: {e}")


#_____________________________________________________________________________________________
# الواجهة الثالثة
elif pages == "📊 تنبأ لأكثر من اسرة واحدة":
    st.set_page_config(page_title="تنبؤ دفعي (ملف Excel/CSV)", page_icon="📊", layout="wide")

    st.title("📊 تنبؤ لعدد كبيرة من الأسر على ملف Excel/CSV")
    st.caption("سنضيف عموداً جديداً بالتنبؤات مع خيار تنزيل الملف النهائي.")

    #-----------------------------------------
    # تحميل النموذج من ملف محلي
    #-----------------------------------------
    MODEL_PATH = "model/gboosting_model 25-9-2025.joblib"

    @st.cache_resource
    def load_local_model(path):
        try:
            obj = joblib.load(path)
            # إذا حفظت قاموس يحتوي pipeline
            if isinstance(obj, dict) and "pipeline" in obj:
                model = obj["pipeline"]
                feature_order = obj.get("feature_order", None)
            else:
                model = obj
                feature_order = None
            return model, feature_order
        except Exception as e:
            st.error(f"تعذر تحميل النموذج: {e}")
            return None, None

    model, feature_order = load_local_model(MODEL_PATH)
    if model is None:
        st.stop()
    st.success("تم تحميل النموذج بنجاح ✅")

    #-----------------------------------------
    # رفع ملف البيانات
    #-----------------------------------------
    uploaded = st.file_uploader("ارفع ملف البيانات (Excel/CSV)", type=["xlsx", "xls", "csv"], key="uploader_data")

    if uploaded is None:
        st.stop()

    try:
        df = read_table(uploaded)
    except Exception as e:
        st.error(f"تعذر قراءة الملف: {e}")
        st.stop()

    st.write("حجم البيانات:", df.shape)
    st.dataframe(df.head(20))

    #-----------------------------------------
    # اختيار الأعمدة
    #-----------------------------------------
    expected = feature_names_from_model(model)
    st.subheader("🔧 اختيار الأعمدة المستخدمة في التنبؤ")
    default_cols = [c for c in df.columns if c in expected] if expected is not None else list(df.columns)
    help_txt = "تم اقتراح الأعمدة المتوقعة من النموذج. يمكنك تعديلها." if expected is not None else "سنستخدم جميع الأعمدة افتراضياً."
    selected_cols = st.multiselect("الأعمدة الداخلة إلى النموذج:", options=list(df.columns), default=default_cols, help=help_txt)

    if not selected_cols:
        st.warning("الرجاء اختيار أعمدة على الأقل.")
        st.stop()

    X = df[selected_cols].copy()
    schema = [{"name": c, "type": "numeric" if pd.api.types.is_numeric_dtype(X[c]) else "text"} for c in selected_cols]
    X_cast = cast_inputs(X, schema)

    #-----------------------------------------
    # إعدادات التنبؤ
    #-----------------------------------------
    pred_col_name = st.text_input("اسم عمود المخرجات", value="prediction")
    add_proba = st.checkbox("إضافة أعمدة الاحتمالات (predict_proba)", value=True)

    #-----------------------------------------
    # زر تنفيذ التنبؤ
    #-----------------------------------------
    if st.button("تنفيذ التنبؤ وإضافة العمود ✅", use_container_width=True):
       try:
          def prepare_input(user_df, feature_order):
              missing_cols = [col for col in feature_order if col not in user_df.columns]
              for col in missing_cols:
                  user_df[col] = np.nan
                # عرض الأعمدة الناقصة للمستخدم (اختياري)
              if missing_cols:
                    st.warning(f"تمت إضافة الأعمدة الناقصة التالية كـ NaN: {missing_cols}")
                # إعادة الترتيب
              return user_df[feature_order]

    # جهز البيانات
          X_ready = prepare_input(X_cast, feature_order)

        # نفذ التنبؤ
          out = predict_with_model(model, X_ready)

          y_pred = out["y_pred"]
          out_df = df.copy()
          out_df[pred_col_name] = y_pred

          if add_proba and "y_proba" in out:
              proba = out["y_proba"]
              class_names = out.get("class_names")
              if class_names is None:
                  class_cols = [f"proba_{i}" for i in range(proba.shape[1])]
              else:
                  class_cols = [f"proba_{str(c)}" for c in class_names]
              proba_df = pd.DataFrame(proba, columns=class_cols, index=out_df.index)
              out_df = pd.concat([out_df, proba_df], axis=1)

          st.success("تم حساب التنبؤات وإضافتها للجدول ✅")
          st.dataframe(out_df.head(50))

       except Exception as e:
            st.error(f"❌ خطأ أثناء التنبؤ: {e}")


        # تنزيل كملف Excel
            output_buf = io.BytesIO()
            with pd.ExcelWriter(output_buf, engine="openpyxl") as writer:
               out_df.to_excel(writer, index=False, sheet_name="predictions")
            output_buf.seek(0)
            st.download_button(
              label="⬇️ تنزيل الملف (Excel)",
              data=output_buf,
              file_name="predictions_with_output.xlsx",
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
              use_container_width=True
        )
       except Exception as e:
          st.error(f"حصل خطأ أثناء التنبؤ: {e}")
