# Real DRL+GWO simulation for the Heliyon revision

هذه الحزمة تنفيذ **حقيقي وكامل** يحل محل الكود الأصلي (الذي كان يولّد النتائج من صيغ
مغلقة بدل محاكاة فعلية). كل رقم يخرج من هنا هو رقم مقاس من تشغيل بيئة MDP فعلية.

## التثبيت

```bash
python3 -m venv venv
source venv/bin/activate        # على ويندوز: venv\Scripts\activate
pip install -r requirements.txt
```

لا حاجة لـ GPU؛ الشبكات صغيرة (128 وحدة مخفية) والتشغيل على CPU كافٍ.

## التشغيل

```bash
python run_experiment.py
```

هذا يُكمل **تلقائياً** من حيث توقفت آخر مرة (يقرأ `results/results.csv` الموجود
حالياً في الحزمة، وهو 64 نتيجة حقيقية من تشغيل سابق تم توليدها فعلياً قبل انقطاع
الجلسة — RR/WF/GRD/PF وPPO/TD3/SAC/SAC+EHS/SAC+GWO كاملة، وPROPOSED بذرتان فقط).
إن انقطع التشغيل لأي سبب، شغّل نفس الأمر مرة أخرى وسيتجاهل كل ما اكتمل ولن يكرره
ولن يهدر وقتاً.

**الوقت المتوقع للمتبقي**: على معالج عادي بنواة واحدة، ما تبقى (استكمال
PROPOSED‑NG/PROPOSED/PROPOSED+ الرئيسية، قابلية التوسّع N=50/70/90، تحليل
الحساسية، وتجربة cold-start) يستغرق تقريباً **2–4 ساعات**. يمكن تشغيله في الخلفية:

```bash
nohup python run_experiment.py > run.log 2>&1 &
tail -f run.log        # لمتابعة التقدّم
```

## بعد الانتهاء

```bash
python analyze.py      # يطبع جدول النتائج + اختبارات الدلالة + تحليل cold-start
python make_figs.py    # يولّد كل الأشكال بدقة 300 dpi في results/figs/
```

`analyze.py` يُخرج أيضاً:
- `results/summary_main.csv` — mean±std لكل طريقة
- `results/significance.csv` — Welch's t-test + Cliff's delta + تصحيح Holm،
  بالمقارنة مع PROPOSED+
- `results/coldstart_summary.csv` — أداء الموثوقية في الحلقات الأولى فقط

## بنية الكود (`sixg/`)

| ملف | المحتوى |
|---|---|
| `env.py` | بيئة 6G: طوابير حزم بمواعيد نهائية، قناة UMi مترابطة زمنياً، goodput، تحكّم فعلي بالقدرة |
| `heuristics.py` | Water-Filling وRound-Robin وGreedy وProportional-Fair و**GWO نخبوي حقيقي** (Mirjalili 2014) |
| `agents.py` | SAC (twin critics + auto-α) وTD3 (delayed updates) وPPO (GAE-λ) — سعة شبكية وميزانية تدريب متطابقة للجميع |
| `runner.py` | حلقة التدريب/التقييم لكل طريقة، بما فيها متغيرات المقترح |

## الطرائق المُقارَنة

- **الاستدلاليات**: RR, WF, GRD, PF (بلا تعلّم)
- **خطوط الأساس DRL**: SAC, TD3, PPO (حقيقية بالكامل، لا صيغ معايرة لتخسر)
- **Ablation**: SAC+EHS (بذر فقط، بلا GWO)، SAC+GWO (GWO من بذور عشوائية، بلا EHS)
- **المقترح**: PROPOSED (EHS + GWO + بوّابة تكيّفية)، PROPOSED-NG (بلا بوّابة)،
  **PROPOSED+** (يضيف "البذر الزمني" — إعادة استخدام أفضل حل من الفتحة السابقة،
  مبرَّر بترابط القناة الزمني ρ=0.9؛ هذا هو التحسين الخوارزمي المضاف)
- **PROPOSED+NOEHS** (المراجعة الثانية، المقيّم #3): ضابط مطابق لـ PROPOSED+ — نفس البوابة
  والبذر الزمني وSAC والبيئة، والفرق الوحيد أن بذور GWO عشوائية بدل مراسي EHS.
  يُشغَّل بـ `python run_r2_coldstart_matched.py` (10 بذور × 25 حلقة) ثم
  `python make_fig_r3_matched.py`؛ المخرجات في `results/coldstart_matched*`.
  (Welch p=0.0002، Cliff's δ=-0.86 على PDR الحلقات 1–5)

## ⚠️ إن جاءت النتيجة مخالفة للمتوقع

إن أظهر أي تحليل أن المقترح **لا يتفوّق** على مقياس معيّن (كما حدث فعلاً مع
Jain's fairness الذي فاز به RR في التشغيل الجزئي) — هذا **لا يُصلَح بتعديل الكود
ليصبح النتيجة مختلفة**. يُكتب كما هو في الورقة كمقايضة صريحة. هذا بالضبط ما
يميّز نتيجة يمكن الدفاع عنها أمام سبعة مقيّمين عن نتيجة يسقطها أول من يفتح الكود.
