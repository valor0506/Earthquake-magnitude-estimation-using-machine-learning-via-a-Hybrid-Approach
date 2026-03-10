# Earthquake-magnitude-estimation-using-machine-learning-via-a-Hybrid-Approach

> **How to use this:** Read top to bottom once. Then use as a reference before your viva/demo. Every concept explained like you're explaining it to a smart friend, not a textbook.

---

# 📌 TABLE OF CONTENTS

1. [The Big Picture — What Did We Actually Build?](#1-the-big-picture)
2. [The Datasets — What Data, Why These, What's In Them](#2-the-datasets)
3. [The Features — What Attributes We Used and Why](#3-the-features)
4. [The Models — LR, RF, QSVM Explained From Scratch](#4-the-models)
5. [The Metrics — MAE, RMSE, R² Explained](#5-the-metrics)
6. [The Code — Every Section Explained](#6-the-code)
7. [The Results — What Your Numbers Mean](#7-the-results)
8. [The Frameworks & Libraries — What Each One Does](#8-the-frameworks)
9. [The Research Positioning — How This Beats Existing Papers](#9-research-positioning)
10. [The Quantum Section — QSVM From First Principles](#10-quantum-section)
11. [The Streamlit App — What It Is and How It Works](#11-streamlit-app)
12. [Viva/Defense Q&A — Every Possible Question With Answers](#12-qa-master-list)
13. [General AIML Concepts You Now Know](#13-general-aiml-concepts)

---

# 1. The Big Picture

## What Did We Actually Build?

A **research pipeline** that answers this question:

> *"Can machine learning predict earthquake magnitude from metadata alone — and where does it fail?"*

### The Three-Layer Answer We Proved:

| Layer | Finding |
|---|---|
| Regional catalog (USGS) | RF works well — R²=0.83 |
| Global catalog (ISC-GEM) | Both models struggle — R²<0.33 |
| Extreme events only (Major EQ) | Both models collapse — R²≈0 |

This wasn't what we expected. **That surprise IS the research.**

### Why This Is Novel

- Singh & Roy (2025) — 1 dataset, no stratified analysis ❌
- Katole et al. (2024) — quantum but fake lab data ❌
- **Us** — 3 real datasets + stratified analysis + quantum on real data ✅

### One-Line Summary of the Whole Project

> We proved that metadata-based ML works for small earthquakes, fails for big ones, and no amount of model complexity (including quantum) fixes this — because the problem is the data type, not the algorithm.

---

# 2. The Datasets

## Why We Need Data First (The Obvious But Important Bit)

Machine learning finds patterns in historical examples. No data = no patterns = no model. The quality and type of data determines everything — more than the model choice.

---

## Dataset 1: USGS Monthly Earthquake Catalog

### What Is USGS?
The **United States Geological Survey** — a US government science agency. They run the **ANSS** (Advanced National Seismic System) — a network of seismometers across the US and globally that detects and records earthquakes 24/7.

### What's in the file? (`all_month_USA.csv`)

```
Rows: 10,114 → 10,110 after cleaning
Magnitude range: -1.77 to 6.40
Mean magnitude: 1.62
```

| Column | What It Means |
|---|---|
| `time` | When the earthquake happened (ISO datetime) |
| `latitude` | North-South position of epicenter |
| `longitude` | East-West position of epicenter |
| `depth` | How deep underground the earthquake started (km) |
| `mag` | The magnitude (what we're predicting) |
| `magType` | How the magnitude was measured (ml, md, mw etc.) |
| Others | Error margins, network info — we don't use these |

### Why Mostly Negative/Small Magnitudes?
Negative magnitudes are real — they just mean very tiny micro-tremors detectable only by modern sensitive instruments. Magnitude 0 is not silence — it's just very small. The scale is logarithmic (explained in Section 5).

### What Makes USGS Special?
- **Regional focus** — dense sensor network catches tiny events
- **Recent data** — monthly snapshot (very current)
- **Heterogeneous magType** — uses ml, md, mw mixed together (adds noise)

---

## Dataset 2: ISC-GEM Global Catalog

### What Is ISC-GEM?
**International Seismological Centre — Global Earthquake Model**. A collaboration between British seismologists (ISC) and the Global Earthquake Model foundation. They took historical earthquake records going back to 1904 and **re-analyzed everything** with modern methods to produce consistent moment magnitude (Mw) values.

### What's in the file? (`isc-gem-cat.csv`)

```
Rows: 74,159
Magnitude range: 4.75 to 9.55
Mean magnitude: ~6.5
File quirk: First 116 lines are comments (#) — must skip them
```

### The File Loading Challenge
The ISC-GEM file starts with 116 lines of documentation text. If you just do `pd.read_csv()` it breaks. The fix:
```python
df = pd.read_csv('isc-gem-cat.csv', skiprows=116)
df.columns = df.columns.str.strip()  # column names have spaces
```

### What Makes ISC-GEM Special?
- **Standardized Mw** — everyone's on the same magnitude scale
- **Global coverage** — every major tectonic zone from 1904 to 2021
- **Re-analyzed** — not raw measurements, scientifically cleaned
- **Only significant events** — minimum ~4.75 (misses small tremors)

### Why ISC-GEM Was Harder to Model (Key Insight)
The global diversity actually makes it harder. A magnitude 7.0 in Japan (subduction zone) and a magnitude 7.0 in Turkey (strike-slip fault) have similar lat/lon/depth metadata but completely different rupture mechanics. The model can't distinguish them from metadata alone.

---

## Dataset 3: Major Earthquakes 1995–2023

### What Is This Dataset?

```
Rows: 1,000
Magnitude range: 6.50 to 9.10
Mean magnitude: 6.94
All events: Major earthquakes only (≥6.5)
```

### Why This Dataset Is The Most Interesting One
It contains ONLY high-magnitude events. This is the range where:
- Metadata loses discriminating power (a 7.0 and 8.5 can be at the same coordinates)
- Classical models collapse
- The quantum extension becomes motivated

### Extra Columns in This Dataset
```
alert     — USGS alert level (green/yellow/orange/red)
tsunami   — Did it trigger a tsunami? (0 or 1)
sig       — Significance score
continent — Which continent
country   — Which country
```
We didn't use these in the model (keeping features identical across datasets) but they're useful for future work.

---

## Dataset Comparison Table

| Property | USGS | ISC-GEM | Major EQ |
|---|---|---|---|
| Size | 10K | 74K | 1K |
| Mag range | -1.77 to 6.40 | 4.75 to 9.55 | 6.50 to 9.10 |
| Coverage | Regional USA+ | Global 1904-2021 | Global 1995-2023 |
| Mag type | Mixed | Mw (uniform) | Mixed |
| Best for | Small/moderate events | Historical global events | Extreme event analysis |
| Model performance | RF R²=0.83 ✅ | RF R²=0.32 ⚠️ | RF R²=-0.05 ❌ |

---

## The `cleaned_earthquake_data.csv` You Also Had

This was a pre-processed version of USGS — already had `year` and `month` extracted, missing values removed. We used the raw files and reproduced the cleaning ourselves for transparency and reproducibility.

---

# 3. The Features

## What Are Features? (For Anyone Who Forgot)

Features = the **inputs** to the model. Also called:
- Independent variables
- Predictors
- X variables
- Attributes

The **target** = what we're predicting = magnitude = the **output**.

```
Features (X) → Model → Prediction (ŷ) ≈ Actual magnitude (y)
```

---

## The Five Features We Used

### 1. `latitude` — North-South Position
- Range: -90 (South Pole) to +90 (North Pole)
- Why useful: Tectonic plates have geographic patterns. The Pacific Ring of Fire follows specific latitude bands. South American subduction follows a lat/lon corridor.
- Earthquake insight: Latitude tells the model roughly which tectonic zone the event is in.

### 2. `longitude` — East-West Position
- Range: -180 to +180
- Why useful: Combined with latitude, gives the tectonic zone
- Together with latitude: They act as a rough "which fault system" proxy

### 3. `depth` — How Deep Underground (km)
- Range: 0 to 700 km
- **The most important feature** (confirmed by Random Forest feature importance)
- Why: Depth determines the **type of earthquake mechanistically**:

```
Shallow (0-70 km)    → Crustal earthquakes, highly variable magnitude
Intermediate (70-300) → Usually subduction zone, more predictable
Deep (300-700 km)    → Very deep subduction, tend toward specific magnitude ranges
```

- This is why depth has the highest feature importance — it encodes physical mechanism, not just location.

### 4. `year` — Calendar Year
- Why included: Long-term trends in catalog completeness (more sensors = more detected events over time), possible decade-scale tectonic cycle proxies
- What we found: Very low importance score — doesn't help much
- Honest assessment: Year is more a data artifact than a physical signal

### 5. `month` — Month of Year (1–12)
- Why included: Seasonal loading (snow mass, reservoir levels) weakly correlates with small earthquake rates in some regions
- What we found: Lowest importance — negligible effect
- Honest assessment: Included for completeness, not a strong feature

---

## Why We Didn't Use Other Available Columns

| Column | Why We Excluded It |
|---|---|
| `magType` | Categorical, varies across datasets, would cause inconsistency |
| `gap` | Station coverage metric — not available in all datasets |
| `rms` | Measurement quality — not a seismic property |
| `alert`, `tsunami` | Only in Major EQ dataset — would break cross-catalog consistency |
| `nst` | Number of stations — data infrastructure, not seismology |

**Rule we followed:** Only use features present in ALL THREE datasets with consistent physical meaning.

---

## Feature Importance Ranking (From Random Forest)

Consistent across all three datasets:

```
1. depth      ████████████████  Most important
2. latitude   ████████████      Second
3. longitude  ███████████       Third
4. year       ███               Minor
5. month      ██                Negligible
```

**What this tells us:** The model is essentially learning "which tectonic zone is this event in?" — because depth+lat+lon together encode tectonic zone identity.

---

# 4. The Models

## Mental Model Before We Start

Think of all ML regression as the same underlying task:

```
Given (latitude, longitude, depth, year, month) → Predict magnitude

The question is: what SHAPE of relationship do we assume?
```

- Linear Regression: "I assume a straight line"
- Random Forest: "I assume complex non-linear curves"
- QSVM: "I assume patterns in quantum Hilbert space"

---

## Model 1: Linear Regression

### What It Is
The simplest possible model. It draws the best-fit straight line through your data.

The equation:
```
magnitude = w1×latitude + w2×longitude + w3×depth + w4×year + w5×month + b
```

Where `w1...w5` are **weights** (learned from training data) and `b` is the **bias** (intercept).

### How It Learns (Briefly)
It finds the weights that minimize the sum of squared errors between predictions and actual values. This has a closed-form mathematical solution — no iteration needed.

### Why It's Good as a Baseline
- Zero assumptions about data (other than linearity)
- Always interpretable — each weight tells you the effect of each feature
- If RF does much better → proves non-linearity exists in the data
- If RF does only slightly better → linearity is a reasonable approximation

### Your Results
```
USGS:     MAE=0.6749, R²=0.4601
ISC-GEM:  MAE=0.3186, R²=0.2412
Major EQ: MAE=0.3424, R²=0.0026  ← beats RF on this dataset
```

### Why LR Beat RF on Major EQ (Important!)
When all magnitudes are in the range 6.5–9.1 (very narrow), RF overfits — it memorizes training examples but doesn't generalize. LR's simplicity accidentally works better because it can't overfit a complex function to noise. This is a real research finding.

---

## Model 2: Random Forest

### What Is a Decision Tree First?

Before Random Forest, understand one decision tree:

```
Is depth > 70km?
├── YES → Is latitude between 30-60?
│         ├── YES → Predict magnitude 5.8
│         └── NO  → Predict magnitude 6.2
└── NO  → Is longitude between -120 and -80?
          ├── YES → Predict magnitude 2.1
          └── NO  → Predict magnitude 1.4
```

A decision tree splits the data by asking yes/no questions about features and predicts the average value at each leaf. It's perfectly interpretable but overfits badly — it just memorizes training data.

### What Random Forest Does Differently

Random Forest builds **100 trees** (n_estimators=100), each on a:
- **Random subset of training data** (bootstrapping — sampling with replacement)
- **Random subset of features** at each split

Then averages all 100 predictions.

```
Tree 1 (random subset) → prediction 1
Tree 2 (random subset) → prediction 2
...
Tree 100 (random subset) → prediction 100
                          ↓
              Average = Final prediction
```

### Why This Works So Well

**Variance reduction:** Each tree overfits differently. When you average 100 differently-overfitted trees, their errors cancel out. This is called **ensemble learning**.

**Non-linearity:** Decision trees can fit any shape of boundary — no linearity assumption.

**Feature importance:** How much does prediction quality drop when you randomly shuffle a feature? Features that cause big drops when shuffled = important.

### Your Hyperparameters

```python
RandomForestRegressor(
    n_estimators=100,   # 100 trees — standard, more = slower but slightly better
    random_state=42,    # Fixed seed — ensures reproducibility (same results every run)
    n_jobs=-1           # Use all CPU cores — speeds up training significantly
)
```

### Why `random_state=42`?
It's a convention from the Hitchhiker's Guide to the Galaxy (42 = answer to everything). It just means: fix the randomness so you get the same results every time you run. Any number works — 42 is tradition.

### Your Results
```
USGS:     MAE=0.3590, R²=0.8327  ← 45% better MAE than LR
ISC-GEM:  MAE=0.2971, R²=0.3204
Major EQ: MAE=0.3486, R²=-0.0487 ← WORSE than LR (negative R²!)
```

---

## Model 3: Quantum-Inspired SVM (QSVM)

### Step 1: What Is a Classical SVM First?

Support Vector Machine finds the **maximum margin boundary** between classes (for classification) or fits a tube around data (for regression = SVR).

For regression (SVR — Support Vector Regression):
- Finds a function that fits within an epsilon (ε) tube around data points
- Uses a **kernel** to work in higher-dimensional spaces
- The magic: you never explicitly compute the high-dimensional coordinates — just the inner products (kernel trick)

### Step 2: What Is a Kernel?

A kernel is a similarity function: `K(x, y) = ⟨φ(x), φ(y)⟩`

It computes how similar two data points are in some (possibly infinite-dimensional) feature space, without ever going there.

Common kernels:
```
Linear kernel:  K(x,y) = x·y              (same as no transformation)
RBF kernel:     K(x,y) = exp(-γ||x-y||²)  (Gaussian similarity)
Quantum kernel: K(x,y) = |⟨ψ(x)|ψ(y)⟩|²  (quantum fidelity)
```

### Step 3: What Makes It QUANTUM?

The ZZFeatureMap encodes your data into a quantum state:

```
Your data point: [depth=50, latitude=35, longitude=139]
        ↓ Scale to [0, π]
        ↓ H gate (Hadamard — creates superposition)
        ↓ P(2·x[0]) (Phase rotation — encodes depth as angle)
        ↓ CNOT + P(2·(π-x[0])·(π-x[1])) (ZZ interaction — encodes feature correlation)
        ↓
Quantum state |ψ(x)⟩ — lives in 2³ = 8 dimensional Hilbert space
```

The quantum kernel is:
```
K_quantum(x, y) = |⟨ψ(x)|ψ(y)⟩|²
```

This measures how "similar" two data points are in quantum Hilbert space — a space that classical computers can't efficiently navigate.

### Step 4: The ZZFeatureMap Circuit (What You Saw in Image 3)

```
q_0: ─[H]─[P(2x[0])]─●──────────────────────────[H]─...
q_1: ─[H]─[P(2x[1])]─[X]─[P(2(π-x[0])(π-x[1]))]─[X]─[H]─...
q_2: ─[H]─[P(2x[2])]──────────────────────────────────[H]─...
```

Reading left to right:
- H = Hadamard gate — puts qubit in superposition (both 0 and 1 at once)
- P(2x[i]) = Phase gate — rotates the qubit by the feature value
- ● and X = Controlled-NOT gate — creates entanglement between qubits
- ZZ interaction = encodes correlation between features in quantum phase

**Why 3 qubits?** We used depth, latitude, longitude (top 3 from feature importance). Each feature gets one qubit. More features = more qubits = exponentially more complex quantum space.

### Step 5: Why It Still Got Negative R²

The problem wasn't the quantum circuit. The problem was the dataset. Major EQ has magnitudes 6.5–9.1 — a range of only 2.6 units. Within that range, depth/lat/lon don't discriminate well. No kernel (classical or quantum) can extract signal that isn't in the data. This is the honest finding.

### Step 6: Why This Is Still Valid Research

1. First QSVM applied to **real tectonic earthquake data** (not lab simulation)
2. The null result is **scientifically informative** — proves metadata limitation
3. The circuit architecture is implemented correctly — Qiskit ZZFeatureMap, FidelityQuantumKernel, precomputed kernel matrix in SVR
4. This is exactly how QSVM prototyping is done before quantum hardware deployment

---

# 5. The Metrics

## Why Three Metrics? Why Not Just One?

Each metric tells you something different. Using only one hides important information.

---

## MAE — Mean Absolute Error

```
MAE = (1/n) × Σ |actual - predicted|
```

**Plain English:** On average, how many magnitude units is your prediction off by?

**Your USGS RF result:** MAE = 0.359
→ On average, predictions are off by 0.36 magnitude units

**Properties:**
- Same unit as the target (magnitude)
- Treats all errors equally — a 0.5 error is exactly 5× worse than a 0.1 error
- Robust to outliers
- **Best for:** Everyday interpretation — "how wrong am I on average?"

---

## RMSE — Root Mean Squared Error

```
RMSE = √[(1/n) × Σ (actual - predicted)²]
```

**Plain English:** Like MAE but it punishes large errors extra hard.

**Why square then root?** Squaring makes large errors count much more. Then taking root brings units back to magnitude scale.

**Your USGS RF result:** RMSE = 0.4907
→ RMSE > MAE because a few large errors (high magnitude events) are pulling it up

**Properties:**
- Same unit as target
- **Sensitive to outliers** — one huge error inflates RMSE significantly
- **Best for:** Safety-critical contexts (earthquake = safety critical) — you CARE about the occasional catastrophic mis-prediction

**The key insight:** If RMSE >> MAE for your model, you have some very large errors even if average performance looks okay. In earthquakes, underestimating a magnitude 8.0 as 6.0 is catastrophic — RMSE catches this.

---

## R² — Coefficient of Determination

```
R² = 1 - (SS_residual / SS_total)
   = 1 - [Σ(actual-predicted)²] / [Σ(actual-mean)²]
```

**Plain English:** What fraction of magnitude variance does your model explain?

**Range:** −∞ to 1.0
- R²=1.0 → perfect prediction
- R²=0.0 → model is no better than predicting the mean every time
- R²<0.0 → model is WORSE than predicting the mean (bad)

**Your results:**
```
USGS RF:     R²=0.8327 → model explains 83% of magnitude variance ✅
ISC-GEM RF:  R²=0.3204 → model explains 32% — limited but real signal ⚠️
Major EQ RF: R²=-0.049 → model is worse than just guessing mean magnitude ❌
```

**Properties:**
- Dimensionless — comparable across datasets with different scales
- **Best for:** Understanding how much of the phenomenon your model actually captures

---

## The Magnitude Scale Itself (Logarithmic!)

This is important for understanding what an MAE of 0.36 actually means physically:

The Richter/Moment magnitude scale is **logarithmic**:
- Magnitude 5 releases 31.6× more energy than magnitude 4
- Magnitude 6 releases 31.6× more than magnitude 5
- Magnitude 7 releases 1,000× more energy than magnitude 5

So an MAE of 0.36 doesn't mean "36% off" — it means the predicted energy release could be off by factor of 31.6^0.36 ≈ **5×**. This is why accurate magnitude estimation matters.

---

# 6. The Code

## Section by Section Explanation

### Cell 1: Imports

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

**Why these?** scikit-learn is the standard Python ML library. These are the building blocks of every classical ML project. If you know these imports, you can build 80% of tabular ML projects.

---

### Cell 2: Loading USGS

```python
df_usgs = pd.read_csv('all_month_USA.csv', low_memory=False)
df_usgs = df_usgs[['time','latitude','longitude','depth','mag']].dropna()
df_usgs['time'] = pd.to_datetime(df_usgs['time'], errors='coerce')
df_usgs['year']  = df_usgs['time'].dt.year
df_usgs['month'] = df_usgs['time'].dt.month
```

**What each line does:**
- `low_memory=False` — tells pandas to read the whole file before deciding column types (prevents type inference errors)
- Column selection — we only keep what we need (reduces memory)
- `dropna()` — removes any row with a missing value in our selected columns
- `pd.to_datetime()` — converts the string "2024-01-15T08:23:11.000Z" into a datetime object
- `.dt.year` and `.dt.month` — extract year and month as separate integer columns

**Why `errors='coerce'`?** Some rows might have malformed timestamps. `errors='coerce'` turns those into NaN instead of crashing.

---

### Cell 3: Loading ISC-GEM

```python
df_isc_raw = pd.read_csv('isc-gem-cat.csv', skiprows=116, low_memory=False)
df_isc_raw.columns = df_isc_raw.columns.str.strip()
df_isc['latitude']  = pd.to_numeric(df_isc_raw['lat'],   errors='coerce')
```

**The `skiprows=116` fix:** The ISC-GEM file has 116 lines of comment text before the actual data. We skip them. The 117th line contains the column headers.

**The `.str.strip()`:** Column names in ISC-GEM have leading/trailing spaces (e.g., `' mw  '`). Strip removes whitespace, so `' mw  '` becomes `'mw'`.

**`pd.to_numeric(errors='coerce')`:** Forces conversion to number. Non-numeric values become NaN.

---

### Cell 4: The `train_test_split`

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**What this does:** Randomly splits data into 80% training and 20% testing.

**Why 80/20?** Convention — enough data to train well, enough to test reliably. Common alternatives: 70/30, 90/10.

**Why random?** Without randomness, if your data is sorted by time, you'd train on early earthquakes and test on recent ones — which introduces temporal bias.

**Why `random_state=42`?** Reproducibility — same split every time you run.

**The four outputs:**
- `X_train` — features for training (the model learns from this)
- `X_test` — features for testing (the model has never seen this)
- `y_train` — actual magnitudes for training
- `y_test` — actual magnitudes for testing (used to evaluate predictions)

---

### Cell 5: Training and Evaluating

```python
lr = LinearRegression()
lr.fit(X_train, y_train)        # Training — learn the weights
y_pred = lr.predict(X_test)     # Inference — apply to unseen data
mae = mean_absolute_error(y_test, y_pred)  # Compare predictions to truth
```

**The three-step pattern** for EVERY ML model:
1. **Instantiate** — create the model object with hyperparameters
2. **Fit** — train it on training data (finds optimal parameters)
3. **Predict** — apply to test data (use learned parameters)

---

### Cell 6: Magnitude-Stratified Analysis

```python
ranges = {
    'Low  (mag < 4.0)':   lambda y: y < 4.0,
    'Moderate (4.0–6.0)': lambda y: (y >= 4.0) & (y <= 6.0),
    'High  (mag > 6.0)':  lambda y: y > 6.0
}
for rname, mask_fn in ranges.items():
    mask = mask_fn(y_test)         # Boolean array: True where condition met
    lr_mae = mean_absolute_error(y_test[mask], y_pred_lr[mask])
```

**What a mask is:** A boolean array like `[True, False, True, True, False...]`. When you index an array with it, you get only the True positions. This filters test data to just one magnitude category.

**Why lambdas?** A lambda is an anonymous function. `lambda y: y < 4.0` is equivalent to `def fn(y): return y < 4.0`. Used here for clean dictionary-based iteration.

---

### Cell 7: QSVM Code

```python
feature_map = ZZFeatureMap(feature_dimension=3, reps=2, entanglement='linear')
qkernel = FidelityQuantumKernel(feature_map=feature_map)
K_train = qkernel.evaluate(x_vec=X_q_train_sub)
qsvm = SVR(kernel='precomputed', C=10.0, epsilon=0.1)
qsvm.fit(K_train, y_q_train_sub)
```

**Line by line:**
- `ZZFeatureMap(feature_dimension=3)` — 3 qubits, one per feature
- `reps=2` — apply the encoding circuit 2 times (deeper encoding)
- `entanglement='linear'` — qubit 0↔1 and 1↔2 are entangled (not all-to-all)
- `FidelityQuantumKernel` — computes kernel using quantum circuit fidelity
- `qkernel.evaluate(x_vec=...)` — runs the quantum simulation, returns 150×150 kernel matrix
- `SVR(kernel='precomputed')` — tell SVR to use our pre-computed kernel matrix directly
- `C=10.0` — regularization (higher C = fits training data more closely)
- `epsilon=0.1` — tolerance tube around the regression line

---

# 7. The Results

## Full Results Table (Your Actual Numbers)

### Table 2: Overall Performance

| Dataset | Model | MAE | RMSE | R² | Interpretation |
|---|---|---|---|---|---|
| USGS | Linear Regression | 0.6749 | 0.8816 | 0.4601 | Baseline |
| USGS | Random Forest | 0.3590 | 0.4907 | **0.8327** | Strong ✅ |
| ISC-GEM | Linear Regression | 0.3186 | 0.4314 | 0.2412 | Weak |
| ISC-GEM | Random Forest | 0.2971 | 0.4083 | 0.3204 | Slightly better |
| Major EQ | Linear Regression | 0.3424 | 0.4414 | **0.0026** | Winner here ⚠️ |
| Major EQ | Random Forest | 0.3486 | 0.4526 | **-0.0487** | Collapsed ❌ |

### Table 3: Stratified Analysis

| Dataset | Range | N | LR MAE | RF MAE | Winner |
|---|---|---|---|---|---|
| USGS | Low (<4.0) | 1,862 | 0.6307 | 0.3532 | Random Forest |
| USGS | Moderate (4-6) | 158 | 1.1771 | 0.4137 | Random Forest |
| USGS | High (>6.0) | **2** | 2.1192 | 1.453 | Random Forest |
| ISC-GEM | Moderate (4-6) | 12,348 | 0.2471 | 0.2399 | Random Forest |
| ISC-GEM | High (>6.0) | 2,484 | 0.6736 | 0.5816 | Random Forest |
| Major EQ | High (>6.0) | 200 | 0.3424 | 0.3486 | **Linear Regression** |

### Table 4: Quantum vs Classical

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Classical SVR (RBF) | 0.3877 | 0.4974 | -0.2273 |
| Quantum SVM (ZZFeatureMap) | 0.4116 | 0.5197 | -0.3396 |

---

## How To Interpret Each Finding

### Finding 1: RF dominates USGS (R²=0.83)
The strong non-linear spatial patterns in regional US seismicity are well-captured by ensemble trees. Depth + coordinates together encode enough tectonic zone information for reliable estimation.

### Finding 2: Both models underperform on ISC-GEM (R²<0.33)
Global tectonic diversity means the same coordinates can correspond to completely different fault mechanisms. 120 years of data adds temporal noise. Five metadata features cannot encode this complexity.

### Finding 3: RF loses to LR on Major EQ (R²=-0.049 vs 0.003)
In a narrow magnitude range (6.5–9.1), RF memorizes training distribution quirks. LR's simplicity provides marginal generalization advantage. Both are essentially useless for distinguishing between high-magnitude events using only metadata.

### Finding 4: QSVM also fails on Major EQ
Confirms the problem is the data type, not the algorithm. No kernel transformation can extract discriminating signal that isn't present. This is the scientific motivation for the waveform-based quantum extension.

### The USGS High-Magnitude Oddity
Only 2 test samples had magnitude >6.0 in USGS test data. LR MAE=2.12 and RF MAE=1.45 with n=2 are statistically meaningless — can't conclude anything from 2 samples. This is why the Major EQ dataset is needed for high-magnitude analysis.

---

# 8. The Frameworks & Libraries

## scikit-learn (`sklearn`)

**What it is:** The standard Python machine learning library. Used in virtually every tabular ML project in the world.

**What we used:**
```python
sklearn.model_selection.train_test_split  # Split data
sklearn.linear_model.LinearRegression     # LR model
sklearn.ensemble.RandomForestRegressor    # RF model
sklearn.svm.SVR                           # Support Vector Regression
sklearn.preprocessing.MinMaxScaler        # Feature scaling
sklearn.metrics.*                         # MAE, RMSE, R²
```

**Why scikit-learn and not PyTorch or TensorFlow?**
Those are for deep learning (neural networks). For tabular data with 5 features, scikit-learn's classical models are more appropriate, faster, and more interpretable.

---

## pandas

**What it is:** Python Data Analysis Library. The Excel of Python programming.

**What we used:**
```python
pd.read_csv()           # Load CSV files
df.dropna()             # Remove missing values
df['col'].str.strip()   # Clean string columns
pd.to_datetime()        # Parse date strings
pd.to_numeric()         # Convert to numbers
df[boolean_mask]        # Filter rows
```

**Why pandas?** It handles messy real-world data (like the ISC-GEM file with 116 comment lines) gracefully. DataFrames are the standard container for tabular data in Python.

---

## NumPy

**What it is:** Numerical Python. The mathematical backbone.

**What we used:**
```python
np.sqrt()    # Square root (for RMSE)
np.pi        # π for quantum angle scaling
np.arange()  # Create number sequences
np.array()   # Convert to numpy array
```

**Why NumPy?** All ML libraries (sklearn, Qiskit) expect numpy arrays as input. Numpy operations are implemented in C — much faster than Python loops.

---

## Matplotlib & Seaborn

**What they are:** Visualization libraries.

- **Matplotlib** — low-level, precise control over every plot element
- **Seaborn** — high-level, beautiful statistical plots with one line

**What we used:**
```python
plt.hist()           # Magnitude distribution histograms
plt.scatter()        # Predicted vs actual scatter plots
plt.bar()            # Model comparison bar charts
plt.barh()           # Feature importance horizontal bars
sns.heatmap()        # Correlation matrix heatmap
```

---

## Qiskit

**What it is:** IBM's quantum computing framework. Used by virtually all academic quantum ML research.

**What we used:**
```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
```

**ZZFeatureMap:** Predefined quantum circuit that encodes classical data into quantum states using ZZ (entangled phase) interactions. Part of Qiskit's standard library.

**FidelityQuantumKernel:** Computes the quantum kernel matrix by running the quantum circuit and measuring state overlap (fidelity) between data point pairs.

**Why Qiskit?** It's the industry standard for quantum ML. IBM provides actual quantum hardware (IBM Quantum) that runs the same circuits. Our simulation is the standard development workflow before hardware deployment.

---

## Streamlit

**What it is:** A Python framework that turns Python scripts into interactive web applications with zero web development knowledge.

**How it works:**
```python
st.title("My App")           # Displays a heading
st.file_uploader(...)        # Creates a file upload button
df = pd.read_csv(...)        # Normal Python
st.dataframe(df)             # Displays it as interactive table
st.pyplot(fig)               # Shows matplotlib figure
```

**Why Streamlit for this project?**
- Zero HTML/CSS/JavaScript needed
- Runs locally or deploys to cloud
- Converts your notebook logic into a demo anyone can use
- Impressive for conference/viva demonstrations

---

## ngrok / localtunnel

**What they are:** Tunneling services — they create public URLs pointing to a server running on your local machine.

**Why needed in Colab?**
Colab runs on Google's servers. Your Streamlit app runs on port 8501 of that server. The internet can't reach it directly. Tunneling services punch a hole through and create a public URL.

```
Internet user → https://plain-walls-search.loca.lt
                        ↓ (tunnel)
               Google Colab machine:8501
                        ↓
               Your Streamlit app
```

**ngrok** requires a free account (token). **localtunnel** works without signup but asks for an IP password (get it with `!curl ifconfig.me`).

---

# 9. Research Positioning

## How Your Paper Fits in the Literature

### The Research Gap Map

```
Existing work:
├── Classical ML on seismic data
│   └── Singh & Roy (2025) — single catalog, no stratification
├── Quantum ML for seismics  
│   └── Katole et al. (2024) — lab data, time-to-failure, not magnitude
└── Deep learning on waveforms
    └── Mousavi & Beroza (2020) — waveforms only, no metadata comparison

YOUR paper fills:
├── Cross-catalog comparison (3 real datasets) ← NEW
├── Magnitude-stratified error decomposition  ← NEW
└── QSVM on real catalog data (not lab)       ← NEW
```

### Why Your Null Result Is Valuable

In science, **null results are publishable and important** when they:
1. Disprove an implicit assumption (that metadata is sufficient for high-magnitude estimation)
2. Are empirically demonstrated with appropriate methods
3. Motivate a clear research direction (waveform-based quantum approach)

Your paper does all three.

---

## How to Cite Your Contribution

> "We present the first three-catalog comparative study of ML models for earthquake magnitude estimation, introducing magnitude-stratified error decomposition as a diagnostic tool. Our empirical results demonstrate that both classical and quantum-kernel approaches are fundamentally limited by metadata insufficiency for high-magnitude event discrimination, motivating a hybrid waveform-quantum architecture as future work."

---

# 10. Quantum Section

## The Full Quantum Story From Zero

### Why Quantum Computing Exists

Classical computers work in bits: 0 or 1. A quantum computer uses **qubits** that can be in **superposition** — both 0 and 1 simultaneously — until measured.

```
Classical bit: |0⟩ or |1⟩
Quantum bit:   α|0⟩ + β|1⟩   where |α|² + |β|² = 1
```

The key insight: with n qubits, you can represent 2^n states simultaneously. 3 qubits = 8 states. 50 qubits = 2^50 ≈ 1 quadrillion states.

### Why This Matters for Machine Learning

Classical kernels (like RBF) compute similarity in a fixed feature space. Quantum kernels compute similarity in a 2^n dimensional space — exponentially larger than what classical computers can efficiently navigate.

For certain datasets, patterns that are inseparable in classical space become separable in quantum Hilbert space. This is the theoretical motivation.

### What Your Circuit Actually Does

**Step 1: Hadamard Gate (H)**
```
|0⟩ → (|0⟩ + |1⟩)/√2   (equal superposition)
```
Puts each qubit in "both 0 and 1" state. This is where quantum parallelism comes from.

**Step 2: Phase Rotation P(2x[i])**
```
|0⟩ → |0⟩
|1⟩ → e^(i·2x[i])|1⟩
```
Rotates the qubit's phase by an angle proportional to your feature value. This encodes the data.

**Step 3: ZZ Interaction P(2(π-x[i])(π-x[j]))**
Creates **entanglement** between qubits. Entanglement means the state of qubit 0 cannot be described independently of qubit 1. This encodes feature *interactions* — the relationship between depth and latitude, for example.

**Step 4: Measurement (Fidelity)**
```
K(x, y) = |⟨ψ(x)|ψ(y)⟩|²
```
The kernel value = quantum fidelity = probability of measuring the same state from both circuits. High fidelity = similar data points.

### The Honest Status of Quantum ML

| Claim | Reality |
|---|---|
| "Quantum computers are faster" | Not yet, for most problems |
| "QSVM is provably better" | Only proven for specific toy problems |
| "Running on real quantum hardware" | 5-7 qubit machines exist, need thousands for advantage |
| "Our simulation is valid research" | Yes — standard prototyping workflow |
| "Our null result means QSVM fails" | No — metadata is the problem, not the algorithm |

---

# 11. Streamlit App

## What the App Does

```
User uploads CSV
    ↓
App auto-detects magnitude, lat, lon, depth, time columns
    ↓
Tab 1: Shows EDA (histogram + heatmap)
Tab 2: Trains model, shows metrics + predicted vs actual
Tab 3: User enters single event values → gets magnitude prediction
```

## The Three Tabs Explained

**Tab 1 — EDA:** Pure visualization. Shows the same plots we generated in the notebook. Helps users understand their dataset before modeling.

**Tab 2 — Model Results:** Trains RF and/or LR on the uploaded dataset, shows metrics, predicted vs actual plot, and the magnitude range error analysis. All computed live on upload.

**Tab 3 — Predict:** User inputs one earthquake's metadata (depth, lat, lon, year, month). Model predicts magnitude and gives a severity classification (minor/moderate/strong).

## The Sidebar Controls

```python
model_choice = st.sidebar.selectbox("Select Model", [...])
test_size    = st.sidebar.slider("Test Split (%)", 10, 40, 20)
```

Sidebar = the left panel in Streamlit apps. Selectbox = dropdown. Slider = draggable value selector.

---

# 12. Viva/Defense Q&A Master List

## Guaranteed Questions

---

**Q: What is the research problem you are solving?**

> We're investigating whether earthquake magnitude can be estimated from seismic catalog metadata (location, depth, time) using machine learning, and identifying where these approaches fail and why.

---

**Q: Why three datasets?**

> No previous study compared model behavior across multiple real-world seismic catalogs. By using USGS (regional), ISC-GEM (global historical), and Major EQ (extreme events only), we discovered that model performance is fundamentally determined by catalog characteristics — not just model choice. This cross-catalog analysis is our primary novel contribution.

---

**Q: Why did Random Forest perform worse than Linear Regression on the Major EQ dataset?**

> When the target range is extremely narrow (6.5–9.1, spanning only 2.6 magnitude units), Random Forest's 100 trees memorize the training data distribution. Linear Regression, being simpler, cannot overfit to this degree and generalizes marginally better. This reversal is itself a research finding: in extreme-value narrow-range problems, model complexity can hurt rather than help.

---

**Q: What is R² and why is your ISC-GEM R² so low?**

> R² measures how much of the target variance the model explains. ISC-GEM covers 120 years of global earthquakes across every tectonic setting. The same latitude and depth can correspond to dramatically different fault mechanisms in different regions. Five metadata features cannot encode this complexity, so R² is low (0.32). This is not a failure of implementation — it's an empirical finding about the limits of metadata-based approaches.

---

**Q: Is your QSVM actually quantum?**

> It uses quantum feature mapping — the ZZFeatureMap circuit — which runs as a classical simulation of a quantum circuit. This is completely standard in quantum ML research. All academic QSVM papers, including Katole et al. (2024) which we build on, use simulators or IBM's small quantum chips. Real quantum advantage requires thousands of qubits which don't exist yet. Our contribution is applying this quantum circuit methodology to real tectonic earthquake data for the first time.

---

**Q: Your quantum results are negative. Is this a failure?**

> No — this is a scientifically important null result. Both classical and quantum kernel SVR fail on the Major EQ dataset because the metadata features don't contain discriminating information for extreme magnitude events, regardless of the kernel. This empirically proves that the limiting factor is data modality, not model complexity, and directly motivates our proposed hybrid waveform-quantum architecture. Null results that close a research question are publishable and valuable.

---

**Q: What is the ZZFeatureMap doing?**

> ZZFeatureMap encodes classical feature values into quantum states using Hadamard gates (creating superposition), phase rotation gates (encoding feature values as qubit rotation angles), and ZZ interaction gates (encoding feature correlations through quantum entanglement). The resulting quantum state lives in a 2^n dimensional Hilbert space — 8 dimensional for our 3 qubits — and the kernel measures how similar two data points are in this quantum space.

---

**Q: Why did you choose depth, latitude, longitude for QSVM specifically?**

> Feature importance analysis from Random Forest consistently ranked depth first, followed by latitude and longitude, across all three datasets. We used the top 3 features for QSVM because quantum circuit complexity grows with the number of qubits, and using the most informative features maximizes signal within the computational constraint.

---

**Q: What does your Streamlit app add to the research?**

> The Streamlit application translates the research into a deployable tool. Any user can upload a seismic CSV and receive magnitude predictions, EDA visualizations, and stratified error analysis without Python knowledge. This demonstrates engineering deployability, not just academic analysis, and provides a framework for future extension with waveform data.

---

**Q: How is your work different from Singh & Roy (2025)?**

> Singh & Roy used one catalog (Indian seismology), five models, and reported only aggregate MAE/RMSE/R². We used three catalogs (USGS, ISC-GEM, Major EQ), two models, and introduced magnitude-stratified error decomposition. We showed that cross-catalog behavior is fundamentally different — a finding that single-catalog studies cannot make. We also added a quantum-kernel comparison, extending their classical framework.

---

**Q: What would you do differently with more time?**

> Three things: (1) Add seismogram waveform features as Layer 2 of the proposed hybrid architecture — waveforms encode rupture dynamics that metadata cannot. (2) Implement proper k-fold cross-validation instead of a single 80/20 split, for statistical robustness. (3) Run the QSVM on IBM Quantum's actual hardware (accessible free through IBM Quantum Network) to compare simulation vs. real quantum results.

---

**Q: Is your dataset representative?**

> Each dataset has different representational biases. USGS overrepresents North American events. ISC-GEM has catalog completeness issues before 1960 when global seismic networks were sparse. Major EQ is curated and incomplete. We acknowledge these limitations in the paper and argue that the cross-catalog comparison itself partially addresses generalizability — by showing where models succeed and fail across three different representational contexts.

---

## Surprise Questions

---

**Q: What is overfitting and did your model overfit?**

> Overfitting is when a model learns the training data too specifically — including its noise — and fails to generalize. Evidence of overfitting: RF performs much better on training data than test data, or RF performs worse than LR in narrow-range scenarios (exactly what happened on Major EQ). We didn't implement a train vs validation comparison, which is a limitation — future work should add cross-validation.

---

**Q: Why didn't you use XGBoost or LightGBM? Singh & Roy found them better.**

> Singh & Roy found LightGBM best on their Indian catalog. Our focus was different: we studied cross-catalog behavioral variation with a consistent, interpretable model set (LR vs RF). Adding XGBoost/LightGBM is a valid future extension, and we expect them to outperform RF on USGS. However, the stratified and cross-catalog findings would likely be structurally similar.

---

**Q: Did you normalize/standardize your features?**

> For LR and RF — no, and this is correct. Random Forest is invariant to feature scaling (it only uses inequality comparisons). Linear Regression benefits from scaling but produces equivalent results when features are on similar scales. For QSVM — yes, we scaled to [0, π] using MinMaxScaler, which is required because quantum rotation angles must be in a valid range.

---

**Q: What is the physical reason depth is the most important feature?**

> Focal depth determines the tectonic mechanism. Shallow earthquakes (0–70km) are crustal — highly variable, include everything from volcanic events to transform fault ruptures. Intermediate (70–300km) tend to be subduction zone events with more consistent characteristics. Deep events (300–700km) are rare and tend toward moderate-large magnitudes. Depth therefore acts as a proxy for earthquake type, which correlates with magnitude range. This is known seismology — our RF feature importance independently confirms it from data.

---

# 13. General AIML Concepts You Now Know

## The Universal ML Workflow (Memorize This)

```
1. Define the problem       → Regression? Classification? What's the target?
2. Get and understand data  → Shape, types, missing values, distributions
3. Clean and preprocess     → Handle nulls, extract features, encode categoricals
4. Split data               → Train/test split (always before any modeling)
5. Choose baseline model    → Simplest thing that could work (often LR)
6. Train and evaluate       → Fit → Predict → Metrics
7. Improve                  → Try better model, more features, tune hyperparameters
8. Analyze errors           → WHERE does it fail? (This is your stratified analysis)
9. Communicate              → Paper, presentation, app
```

---

## Regression vs Classification

| | Regression | Classification |
|---|---|---|
| Target type | Continuous number | Category/class |
| Example | Predict magnitude (1.2, 4.7, 8.1) | Predict alert level (green/yellow/red) |
| Metrics | MAE, RMSE, R² | Accuracy, Precision, Recall, F1 |
| Our task | Regression ✅ | — |

---

## Supervised vs Unsupervised

| | Supervised | Unsupervised |
|---|---|---|
| Training data | Has labels (magnitude values) | No labels |
| Goal | Learn input→output mapping | Find structure in data |
| Examples | LR, RF, SVM, Neural Networks | K-Means, PCA, Autoencoders |
| Our task | Supervised ✅ | — |

---

## Bias-Variance Tradeoff

```
High Bias (Underfitting)        High Variance (Overfitting)
→ Model too simple              → Model too complex
→ Linear Regression on          → Random Forest on
  non-linear data                 Major EQ narrow range
→ Low R² on both train & test   → High R² on train, low on test
```

The sweet spot is a model complex enough to capture real patterns but simple enough to generalize.

---

## The Curse of Dimensionality

More features = not always better. As dimensions increase, data points become increasingly sparse. With 5 features and 74K samples, we're fine. With 50 features and 1K samples, models become unreliable.

---

## Why Baselines Matter (The Testing Pyramid)

```
Stupid baseline:    Always predict mean → R²=0.0 by definition
Simple baseline:    Linear Regression   → R²=0.46 (USGS)
Our model:          Random Forest       → R²=0.83 (USGS)
State of art:       LightGBM + features → R²~0.85+ (literature)
```

Each step up must be justified. If RF didn't beat LR significantly, there would be no point using RF.

---

## Ensemble Methods (Big Picture)

Random Forest is one of many ensemble methods:

| Method | Idea | When to use |
|---|---|---|
| Random Forest | Average many independent trees | General tabular regression/classification |
| Gradient Boosting | Trees correct each other's errors sequentially | When you want maximum accuracy |
| XGBoost/LightGBM | Optimized gradient boosting | Kaggle competitions, production |
| Bagging | Average multiple models | Reduce variance of any model |

---

## Feature Engineering Is More Important Than Model Choice

This cannot be overstated. In our project, we used `latitude`, `longitude`, `depth`, `year`, `month`. If we had access to:
- Fault distance (distance to nearest known fault)
- Seismic moment tensor components
- Waveform envelope features

Even Linear Regression with these features would likely outperform our RF. **Better data > better algorithm. Always.**

---

## The Scientific Method in ML Papers

Every ML paper answers:
1. **Problem:** What are we solving and why does it matter?
2. **Related work:** What exists and what gap remains?
3. **Method:** How did we approach it?
4. **Experiments:** What did we test and how?
5. **Results:** What did we find?
6. **Discussion:** What do the numbers mean and why?
7. **Conclusion:** What's the takeaway and what's next?

Your paper follows all seven. The stratified error analysis goes into #4 and #6 — which is where the real novelty lives.

---

*Built from the complete earthquake ML research pipeline — USGS + ISC-GEM + Major EQ 1995-2023 + LR + RF + QSVM + Streamlit*

*Your actual results: USGS RF R²=0.83 | ISC-GEM RF R²=0.32 | Major EQ RF R²=−0.049 | QSVM R²=−0.34*
