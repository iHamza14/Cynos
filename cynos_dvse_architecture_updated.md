# Cynos: DVSE-Based Dead-Reckoning Architecture

> **Status:** Architecture specification for implementation  
> **Reference model:** Xuan Xiao, Xiaotong Ren, Haitao Li, *An Inertial Sequence Learning Framework for Vehicle Speed Estimation via Smartphone IMU* (arXiv:2505.18490v1, 2025).  
> **Scope:** Reproduce the paper's velocity-estimation core as closely as possible, then add Cynos-specific heading and navigation components.

---

## 1. Executive Summary

The reference paper is a **vehicle speed-estimation** system. Its central architecture is:

```text
1-second IMU statistics
        │
        ├── Noise Compensation Network (NCN)
        │        └── GRU → learned scalar disturbance N
        │
        └── Motion Transformation Network (MTN)
                 └── TCN → phone-to-vehicle rotation Rᵥᵖ
                              │
                              ▼
                   physics-based Δv calculation
                              │
                              ▼
                         Δv = Δv_physics + N
                              │
                              ▼
                         vehicle speed
```

Cynos extends that velocity estimator with a separate **global-heading branch**:

```text
raw gyro
   │
   ▼
causal Gyro TCN
   │
   ▼
yaw-rate estimate
   │
   ▼
heading integration
   │
   ▼
vehicle heading
```

The two branches are then fused to produce local displacement during GNSS blackout:

```text
velocity v + heading ψ
        │
        ▼
     ΔE, ΔN
        │
        ▼
   DR trajectory
```

The map-matching, GNSS handover, and NHC/Viterbi components remain outside the neural models.

---

# 2. Paper vs. Cynos

The most important distinction is:

| Component | Reference paper | Cynos |
|---|---|---|
| 1-second IMU statistics | Yes | Yes |
| Acceleration features | 18 | 18 |
| Gyroscope features | 18 | 18 |
| Reference velocity `Vᵣ` | Yes | Yes |
| NCN | GRU, hidden 128 | Same |
| MTN | TCN | Same |
| MTN output | 3 Euler angles | Same |
| Phone → vehicle rotation | Yes | Yes |
| Physics-based velocity integration | Yes | Yes |
| Learned velocity disturbance `N` | Yes | Yes |
| GNSS loss matching | Yes | Only for GNSS-supervised training |
| Random sensor-frame rotation augmentation | Yes | Yes |
| Global heading estimator | **No** | Gyro TCN |
| Global position propagation | Not the paper's main navigation task | Yes |
| GNSS/DR state machine | No complete Cynos-style module | Yes |
| Viterbi map matching | No | Mobile/navigation layer |
| NHC | No | Mobile/navigation layer |
| Reference deployment | ONNX Runtime on Android | TFLite planned for Cynos |

The key implementation decision is therefore:

> **Do not use the current generic velocity GRU+TCN as the DVSE reproduction. Replace it with the paper's NCN-GRU + MTN-TCN + explicit physics path. Keep the existing heading TCN as a separate Cynos extension.**

---

# 3. System Architecture

```text
                             PHONE IMU
                   ┌─────────────────────────┐
                   │                         │
             Accelerometer               Gyroscope
               10 Hz, 3D                 10 Hz, 3D
                   │                         │
                   │                         ├───────────────────────┐
                   │                         │                       │
                   ▼                         │                       │
        1-second feature extraction          │                Gyro bias correction
          10 samples / second               │                       │
                   │                         │                       ▼
                   │                         │                 Heading TCN
                   │                         │                       │
                   │                         │                       ▼
                   │                         │                 yaw rate r
                   │                         │                       │
                   │                         │                       ▼
                   │                         │                 heading ψ
                   │                         │                       │
          ┌────────┴────────┐                │                       │
          │                 │                │                       │
          ▼                 ▼                │                       │
     Acc features      Gyro features         │                       │
       [10,18]            [10,18]            │                       │
          │                 │                │                       │
          └────────┬────────┘                │                       │
                   │                         │                       │
                   ▼                         │                       │
               NCN (GRU)                    │                       │
                   │                         │                       │
                   ▼                         │                       │
             disturbance N                  │                       │
                                             │                       │
                                             │                       │
                              Acceleration preintegration            │
                                     [10,3]                           │
                                         │                           │
                              + gravity reference                    │
                                     [0,0,9.81]                      │
                                         │                           │
                                         ▼                           │
                                  MTN (TCN)                           │
                                         │                           │
                                         ▼                           │
                                    α, β, γ                           │
                                         │                            │
                                         ▼                            │
                                     Rᵥᵖ                            │
                                         │                            │
                      raw acceleration + gravity removal             │
                                         │                            │
                                         ▼                            │
                                 vehicle forward aᶠ                  │
                                         │                            │
                                         ▼                            │
                               Δv_physics per second                 │
                                         │                            │
                                         +                            │
                                         N                            │
                                         │                            │
                                         ▼                            │
                                  Δv and v[k]                        │
                                         │                            │
                                         └──────────────┬─────────────┘
                                                        │
                                                  v[k] + ψ[k]
                                                        │
                                                        ▼
                                                  ΔE, ΔN
                                                        │
                                                        ▼
                                                DR position
                                                        │
                           ┌────────────────────────────┴────────────┐
                           │                                         │
                      GNSS valid                              GNSS unavailable
                           │                                         │
                      use GNSS                                 use DR trajectory
                           │                                         │
                           └────────────────────┬────────────────────┘
                                                │
                                           navigation layer
                                                │
                                      road graph / NHC / Viterbi
```

---

# 4. Coordinate Systems

Cynos uses three conceptual frames.

## 4.1 Phone frame `p`

The smartphone sensor frame:

```math
a_p =
\begin{bmatrix}
a_x\\
a_y\\
a_z
\end{bmatrix}
```

and

```math
\omega_p =
\begin{bmatrix}
\omega_x\\
\omega_y\\
\omega_z
\end{bmatrix}.
```

The project gyro column convention is:

```text
[yaw, pitch, roll]
```

---

## 4.2 Vehicle frame `v`

The reference paper defines the vehicle forward direction as the positive **Y axis**:

```math
u =
\begin{bmatrix}
0\\
1\\
0
\end{bmatrix}.
```

Therefore the forward acceleration used for the velocity update is:

```math
a_f = a_v^T u = a_y^v.
```

---

## 4.3 Navigation/world frame `w`

For Cynos navigation:

```text
E = East
N = North
heading = 0 at North
heading increases clockwise
```

This navigation convention is a **Cynos system choice**, not a specification from the reference paper.

---

# 5. Sampling and Windowing

Cynos currently uses:

```text
IMU frequency     = 10 Hz
1 second          = 10 samples
context           = 10 seconds
raw context       = 100 samples
```

For one example:

```text
accelerometer = [100, 3]
gyroscope     = [100, 3]
```

Batch form:

```text
accel = [B, 100, 3]
gyro  = [B, 100, 3]
```

The paper also organizes its learning around one-second temporal units, which makes the 10-second Cynos context compatible with the MTN's reported receptive field.

---

# 6. Preprocessing

## 6.1 Gyroscope deterministic bias correction

The current Cynos stationary calibration is:

```text
bias =
[-0.0020,
 +0.0047,
 -0.0022] rad/s
```

Apply:

```math
\omega_{corr} = \omega_{raw} - b_{gyro}.
```

This is a deterministic sensor calibration step, not a learned filter.

For the heading branch, use the corrected gyro signal as the model input.

---

## 6.2 Additional filtering

Do not add an EKF, UKF, Madgwick filter, complementary filter, or arbitrary low-pass stage merely because those are familiar tools.

The current Cynos design intentionally keeps the learned DVSE core explicit:

```text
calibration
   ↓
feature extraction / preintegration
   ↓
NCN + MTN
   ↓
physics
```

Any future filtering stage should be introduced as a measured ablation or engineering requirement, not silently inserted into the baseline.

---

# 7. Data Augmentation

The reference paper applies a random 3-axis rotation to simulate arbitrary phone placement.

Generate:

```math
R_r = R_x(\alpha_r)R_y(\beta_r)R_z(\gamma_r).
```

Apply the **same rotation to the complete IMU window**:

```math
a'_p = R_r a_p
```

```math
\omega'_p = R_r \omega_p.
```

The rotation is sampled **once per window**, not independently per sensor sample.

For a 10-second window:

```text
100 accelerometer samples
100 gyroscope samples
        │
        ▼
    one random Rᵣ
        │
        ▼
all samples transformed by that same Rᵣ
```

This keeps the relative physical relationship between accelerometer and gyroscope measurements intact.

---

# 8. Velocity Branch: Reference Architecture

The velocity branch contains two separate networks:

```text
NCN = Noise Compensation Network
MTN = Motion Transformation Network
```

They solve different problems.

```text
NCN → learns the scalar disturbance / noise correction N

MTN → learns phone-to-vehicle motion transformation Rᵥᵖ
```

This separation is central to the reference design.

---

# 9. NCN Input Features

For every one-second interval, calculate six statistics for each axis:

1. standard deviation
2. maximum
3. minimum
4. RMS
5. skewness
6. kurtosis

For each 3-axis sensor:

```text
3 axes × 6 statistics = 18 features
```

Therefore:

```text
acceleration features = 18
gyroscope features    = 18
combined               = 36
```

For a 10-second sequence:

```text
acc features = [B, 10, 18]
gyro features = [B, 10, 18]
```

### 9.1 Statistics used by Cynos

The paper specifies the six statistics, but does not fully specify every numerical convention for skewness and kurtosis. For implementation, use a deterministic convention.

RMS:

```math
RMS(x)=\sqrt{\frac{1}{n}\sum_i x_i^2}.
```

Skewness:

```math
skew(x)=
\frac{\frac{1}{n}\sum_i(x_i-\mu)^3}
{\sigma^3+\epsilon}.
```

Excess kurtosis:

```math
kurt(x)=
\frac{\frac{1}{n}\sum_i(x_i-\mu)^4}
{\sigma^4+\epsilon}-3.
```

The exact skewness/kurtosis convention above is a **Cynos implementation choice**, not a claim about the paper's hidden implementation.

---

# 10. Reference Velocity `Vᵣ`

The NCN receives a scalar reference velocity.

The reference paper describes:

```text
Training:
    Vᵣ = GNSS speed

Inference:
    Vᵣ = estimated speed
```

The reason is that the disturbance behavior can depend on vehicle speed.

For Cynos:

### At GNSS → DR transition

Initialize from the last trusted GNSS speed:

```text
v0 = last trusted GNSS speed
```

and use:

```text
Vᵣ = v0
```

### During blackout

Do not inject new GNSS speed after blackout starts.

Instead, the recursive DR state provides the reference for later steps.

---

# 11. Noise Compensation Network (NCN)

The reference architecture uses a **one-layer GRU with hidden size 128**.

## 11.1 Input

```text
acc features = [B, 10, 18]
gyro features = [B, 10, 18]
Vᵣ            = [B, 1]
```

For implementation, broadcast the velocity scalar:

```text
Vᵣ sequence = [B, 10, 1]
```

---

## 11.2 Acceleration embedding

```text
18
 ↓
FC
 ↓
32
 ↓
FC
 ↓
64
```

Output:

```text
[B, 10, 64]
```

---

## 11.3 Gyroscope embedding

Same structure:

```text
18
 ↓
FC
 ↓
32
 ↓
FC
 ↓
64
```

Output:

```text
[B, 10, 64]
```

---

## 11.4 Concatenation

```text
64 + 64 = 128
```

So:

```text
[B, 10, 128]
```

---

## 11.5 GRU

```text
input_size  = 128
hidden_size = 128
num_layers  = 1
```

Output:

```text
[B, 10, 128]
```

---

## 11.6 Reference velocity injection

Concatenate the GRU output with `Vᵣ`:

```text
128 + 1 = 129
```

So:

```text
[B, 10, 129]
```

---

## 11.7 Regression head

The reference paper specifies:

```text
129
 ↓
64
 ↓
32
 ↓
16
 ↓
1
```

Therefore:

```text
FC: 129 → 64
FC:  64 → 32
FC:  32 → 16
FC:  16 →  1
```

Output:

```text
N = [B, 10, 1]
```

`N` is a **scalar correction to the one-second velocity increment**. It is not a 3-axis acceleration correction.

---

# 12. NCN Physical Interpretation

The reference formulation separates physical integration from the learned disturbance.

Conceptually:

```text
phone acceleration
       │
       ▼
gravity removal
       │
       ▼
phone → vehicle transform
       │
       ▼
forward-axis projection
       │
       ▼
physics integration
       │
       ▼
Δv_physics
```

The NCN learns:

```text
IMU statistical behavior
        +
reference velocity
        │
        ▼
        N
```

Then:

```math
\boxed{\Delta v = \Delta v_{physics} + N}
```

This is the central mechanism of the paper's velocity estimator.

---

# 13. Motion Transformation Network (MTN)

The MTN estimates the relative rotation between phone and vehicle:

```math
R_v^p.
```

Its purpose is **not** to directly predict speed.

---

# 14. MTN Input

First divide the accelerometer stream into one-second intervals and preintegrate:

```math
I_t = \int_t^{t+1} a_p(\tau)\,d\tau.
```

At 10 Hz:

```text
10 raw acceleration samples
        ↓
1-second integration
        ↓
3 values [Iₓ, Iᵧ, I_z]
```

For 10 seconds:

```text
I = [10, 3]
```

The paper reports this temporal reduction as a way to reduce sequence length for mobile computation.

---

# 15. Gravity Reference

The MTN input also contains the fixed world gravity reference:

```math
g_w =
\begin{bmatrix}
0\\
0\\
9.81
\end{bmatrix}.
```

At each one-second step, concatenate:

```text
[Iₓ, Iᵧ, I_z, 0, 0, 9.81]
```

Therefore:

```text
MTN input = [B, 10, 6]
```

---

# 16. MTN Embedding

The reference architecture uses:

```text
6
 ↓
FC
 ↓
32
 ↓
FC
 ↓
64
```

Output:

```text
[B, 10, 64]
```

---

# 17. MTN TCN

The reference TCN is:

```text
2 temporal blocks
```

Each block contains:

```text
3 dilated causal layers
```

with:

```text
channels = 64
kernel    = 2
```

The dilation pattern is:

```text
Block 1:
    d = 1
    d = 1
    d = 1

Block 2:
    d = 2
    d = 2
    d = 2
```

So the six temporal convolutions are:

```text
64 channels
│
├── Block 1
│   ├── k=2, d=1
│   ├── k=2, d=1
│   └── k=2, d=1
│
└── Block 2
    ├── k=2, d=2
    ├── k=2, d=2
    └── k=2, d=2
```

The MTN output is then mapped:

```text
64
 ↓
FC
 ↓
3
```

giving:

```text
[ B, 10, 3 ]
```

for the Euler angles:

```text
α, β, γ
```

---

# 18. MTN Receptive Field

The reference paper gives:

```math
RF =
1+
N_c(k-1)
\sum_{i=1}^{N_b}2^{i-1}.
```

Using:

```text
N_b = 2
N_c = 3
k   = 2
```

gives:

```math
RF = 1 + 3(1)(1+2)
   = 10.
```

Therefore the MTN sees:

```text
10 one-second steps
```

which matches Cynos's 10-second context.

---

# 19. Euler Angles → Rotation Matrix

The reference paper defines:

```math
R_v^p = R_x(\alpha)R_y(\beta)R_z(\gamma).
```

Use the same rotation order.

```math
R_x(\alpha)=
\begin{bmatrix}
1&0&0\\
0&\cos\alpha&-\sin\alpha\\
0&\sin\alpha&\cos\alpha
\end{bmatrix}
```

```math
R_y(\beta)=
\begin{bmatrix}
\cos\beta&0&\sin\beta\\
0&1&0\\
-\sin\beta&0&\cos\beta
\end{bmatrix}
```

```math
R_z(\gamma)=
\begin{bmatrix}
\cos\gamma&-\sin\gamma&0\\
\sin\gamma&\cos\gamma&0\\
0&0&1
\end{bmatrix}.
```

Do not silently change the rotation order or interpretation when implementing the MTN.

---

# 20. Gravity Removal and Vehicle-Frame Acceleration

The reference paper writes the physical transformation as:

```math
a_v =
R_v^p
\left(
\hat a_p - R_p^w g_w
\right).
```

Here:

```text
R_p^w = world → phone
R_v^p = phone → vehicle
```

and the MTN directly estimates `R_v^p`.

## Cynos engineering closure

For the current implementation, the local world frame used inside the velocity-physics calculation is identified with the vehicle frame. Under that engineering convention:

```math
R_p^w = (R_v^p)^T.
```

Therefore:

```math
g_p = (R_v^p)^T g_w
```

and:

```math
a_v =
R_v^p
\left(
a_p - (R_v^p)^T g_w
\right).
```

**Important:** the world=vehicle identification above is an implementation closure for Cynos. It is not a claim that the reference paper explicitly defines global world and vehicle frames as identical.

After transforming to the vehicle frame, use:

```math
a_f = a_y^v.
```

because vehicle `+Y` is the paper's forward axis.

---

# 21. Physics-Based Velocity Update

For each one-second interval, at 10 Hz:

```math
\Delta v_{physics,k}
=
\sum_{j=1}^{10}
a_{f,k,j}\Delta t
```

with:

```text
Δt = 0.1 s.
```

Pipeline:

```text
10 raw acceleration samples
        ↓
10 forward-acceleration values
        ↓
numerical integration
        ↓
Δv_physics[k]
```

Units:

```text
m/s² × s = m/s
```

Add the learned NCN disturbance:

```math
\boxed{
\Delta v_k =
\Delta v_{physics,k}+N_k
}
```

Then recursively update speed:

```math
v_k = v_{k-1}+\Delta v_k.
```

---

# 22. What the MTN Does Not Predict

The MTN does **not** directly predict velocity.

Its role is:

```text
accelerometer sequence
      ↓
phone/vehicle transformation
      ↓
Rᵥᵖ
```

Velocity is obtained only after:

```text
Rᵥᵖ
 ↓
gravity removal
 ↓
phone → vehicle transform
 ↓
forward-axis projection
 ↓
integration
 ↓
Δv_physics
 ↓
+ N
 ↓
Δv
 ↓
v
```

This is why replacing the MTN with a generic velocity TCN is not a faithful reproduction of the reference architecture.

---

# 23. Heading Branch: Cynos Extension

The reference paper is a speed-estimation paper. It does **not** provide the global vehicle-heading estimator required by Cynos.

The Cynos extension therefore keeps the existing heading branch:

```text
corrected raw gyro
        ↓
causal TCN
        ↓
yaw-rate estimate
        ↓
integration
        ↓
vehicle heading
```

This module is intentionally kept separate from the DVSE velocity core.

---

# 24. Heading TCN Input

Use:

```text
context = 10 seconds
sampling = 10 Hz
samples = 100
gyro axes = 3
```

Input:

```text
[B, 100, 3]
```

Use bias-corrected gyro:

```math
\omega_{corr}[k]
=
\omega_{raw}[k]-b_{gyro}.
```

---

# 25. Heading TCN Architecture

Current Cynos heading model:

```text
Input: [B,100,3]

        ↓
Causal TCN
    channels = 48
    dilation = 1

        ↓
Causal TCN
    channels = 48
    dilation = 2

        ↓
Causal TCN
    channels = 48
    dilation = 4

        ↓
Causal TCN
    channels = 48
    dilation = 8

        ↓
Output: [B,100,1]
```

The output is yaw rate:

```math
\hat r_k \; [deg/s].
```

The target convention already used in the Cynos data is:

```math
r^* = -r_{vehicle}.
```

Do **not** negate the target again downstream.

---

# 26. Heading Integration

Convert yaw rate to radians per second:

```math
\hat\omega_k =
\hat r_k\frac{\pi}{180}.
```

At 10 Hz:

```text
Δt = 0.1 s
```

Per-sample heading increment:

```math
\Delta\psi_k =
\hat\omega_k\Delta t.
```

Integrate:

```math
\psi_k =
wrap(\psi_{k-1}+\Delta\psi_k).
```

The TCN therefore predicts **heading change**, not absolute global heading.

---

# 27. Absolute Heading Initialization

Pure gyro integration cannot determine an absolute north-referenced heading.

At GNSS availability, initialize from GNSS course when the vehicle is moving sufficiently fast for course to be meaningful:

```text
GNSS speed sufficiently high
        ↓
valid GNSS course
        ↓
ψ₀ = GNSS course
```

During blackout:

```math
\psi_{k+1}
=
\psi_k+\Delta\psi_k.
```

When GNSS returns and course is valid, reinitialize the heading state.

---

# 28. Why MTN Euler Angle `γ` Is Not Global Heading

The MTN estimates:

```math
R_v^p
```

which represents the **phone-to-vehicle transformation**.

That is different from:

```text
vehicle orientation relative to North
```

Therefore:

```text
MTN
  → phone/vehicle relative orientation
```

while:

```text
Gyro TCN
  +
GNSS initial course
  →
vehicle heading evolution
```

These are different quantities and should not be merged by simply interpreting `γ` as global yaw.

---

# 29. Velocity + Heading Fusion

After both branches have produced their states:

```text
velocity branch → v[k]    [m/s]
heading branch  → ψ[k]    [rad]
```

At each one-second navigation update:

```math
\Delta s_k = v_k \Delta t
```

with:

```text
Δt = 1 s.
```

Using the Cynos navigation convention:

```math
\Delta E_k
=
\Delta s_k\sin(\psi_k)
```

```math
\Delta N_k
=
\Delta s_k\cos(\psi_k).
```

Then:

```math
E_k = E_{k-1}+\Delta E_k
```

```math
N_k = N_{k-1}+\Delta N_k.
```

The navigation state is therefore:

```text
position
speed
heading
```

---

# 30. Why Fusion Output Is 1 Hz Initially

The DVSE velocity branch naturally operates on one-second steps.

The heading branch operates at 10 Hz, but the first integrated Cynos navigation design should expose the fusion output at 1 Hz:

```text
Velocity inference = 1 Hz
Heading inference  = 10 Hz internally
Fusion output      = 1 Hz
```

The heading branch integrates its ten 10-Hz samples between each pair of one-second fusion boundaries.

Any future 10-Hz position output should be treated as an explicit resampling/interpolation layer rather than implying that the DVSE velocity branch directly estimates 10-Hz velocity.

---

# 31. GNSS → DR Handover

The neural networks should not decide whether GNSS is available. A navigation state machine should.

```text
                  ┌─────────────────┐
                  │   GNSS VALID    │
                  └────────┬────────┘
                           │
                    position = GNSS
                           │
                    reset DR state
                           │
                           ▼
                      GNSS LOST
                           │
                           ▼
                   DEAD RECKONING
                           │
                           ▼
                     GNSS RETURNS
                           │
                           ▼
                    position = GNSS
                    reset DR state
```

When GNSS is valid:

```math
p_{display}=p_{GNSS}.
```

Do not blend GNSS and DR position in the baseline. The Cynos requirement is to trust GNSS when it is valid and use DR only during blackout.

---

# 32. 60-Second Blackout State

At the start of a blackout:

```text
p₀ = last trusted GNSS position
v₀ = last trusted GNSS speed
ψ₀ = last trusted GNSS course
```

Then, once per second:

```text
10-second IMU context
       │
       ├── NCN + MTN
       │        │
       │        ▼
       │       Δv
       │        │
       │        ▼
       │       v[k]
       │
       └── Gyro TCN
                │
                ▼
               Δψ
                │
                ▼
               ψ[k]

v[k] + ψ[k]
       │
       ▼
    ΔE, ΔN
       │
       ▼
    p_DR[k]
```

For a 60-second blackout, evaluate the DR trajectory at:

```text
2 s
5 s
10 s
15 s
30 s
60 s
```

using the project evaluation protocol.

---

# 33. Map Matching and NHC

These are navigation-layer components, not part of the DVSE neural model.

## 33.1 Viterbi map matching

Conceptually:

```text
DR position
   +
heading
   +
speed
   +
road graph
   ↓
road candidates
   ↓
emission costs
   ↓
transition costs
   ↓
fixed-lag Viterbi
   ↓
road-constrained position
```

Keep road graph data out of TFLite.

---

## 33.2 Non-Holonomic Constraint (NHC)

For a road direction `ψ_road`:

```math
v_{lateral}
=
v\sin(\psi-\psi_{road}).
```

This can be used as a lateral-motion consistency term in the navigation layer.

NHC should remain outside the DVSE model initially.

---

# 34. Velocity Training Loss

The reference paper uses SmoothL1 for both the velocity-change and integrated-velocity objectives.

Velocity-change loss:

```math
L_{\Delta v}
=
\frac{1}{n}
\sum_i
SmoothL1(\Delta v_i,\Delta v_i^*).
```

Integrated velocity loss:

```math
L_v
=
\frac{1}{n}
\sum_i
SmoothL1
\left(
\sum_{j\le i}\Delta v_j,
\sum_{j\le i}\Delta v_j^*
\right).
```

The paper uses:

```math
\boxed{
L_{velocity}
=
0.7L_{\Delta v}
+
0.3L_v
}
```

because:

```text
λ = 0.7
```

Keep this weighting for a paper-faithful velocity experiment.

---

# 35. GNSS / IMU Timestamp Matching

The reference paper also evaluates the loss under two temporal alignments:

```text
1. aligned
2. one-second shifted
```

and takes the smaller loss:

```math
L =
\min
\left(
L(x_{2:n},y_{2:n}),
L(x_{1:n-1},y_{2:n})
\right).
```

This is a **GNSS/IMU synchronization strategy from the paper**, not an instruction to shift Cynos labels arbitrarily.

### Cynos rule

The current Cynos training data uses synchronized vehicle ground truth.

Therefore:

```text
vehicle-ground-truth experiments
        ↓
exact timestamps
```

Do not introduce the paper's optional 0/1-second matching into the current vehicle-ground-truth experiment without a specific synchronization problem.

The 0/1-second loss matching can be introduced later when training directly against GNSS.

---

# 36. Heading Training Loss

The global heading branch is a Cynos extension, so its loss is not from the reference paper.

Use a supervised yaw-rate objective:

```math
L_{rate}
=
SmoothL1(\hat r,r^*).
```

Also use an integrated heading-change objective:

```math
L_{\psi}
=
SmoothL1
\left(
\sum_{j\le i}\hat r_j\Delta t,
\sum_{j\le i}r_j^*\Delta t
\right).
```

A starting Cynos objective is:

```math
L_{heading}
=
0.7L_{rate}+0.3L_{\psi}.
```

The `0.7 / 0.3` weighting here is a **Cynos design choice**, not a number taken from the paper.

---

# 37. Overall Training Strategy

Do not immediately train the whole navigation stack end-to-end.

Use three stages.

## Stage A: DVSE velocity core

Train:

```text
NCN
+
MTN
+
physics layer
```

against vehicle speed ground truth.

Output:

```text
10 Δv values
```

then reconstruct:

```text
10 speed values
```

using:

```text
L_velocity = 0.7 L_Δv + 0.3 L_v
```

---

## Stage B: Heading TCN

Train independently:

```text
bias-corrected gyro
        ↓
GyroTCN
        ↓
yaw rate
```

Target:

```math
r^* = -r_{vehicle}.
```

Use a **trajectory/session-separated validation split**.

Do not use the final test set to select checkpoints.

---

## Stage C: Fusion

Initially freeze both trained models:

```text
velocity model → v
heading model  → ψ
```

then evaluate:

```text
v + ψ
 ↓
ΔE, ΔN
 ↓
2/5/10/15/30/60-second blackout metrics
```

Only after this baseline is stable should joint fine-tuning be considered.

---

# 38. Data Split and Evaluation Hygiene

For the current Cynos experiments:

```text
training
validation
final test
```

must remain separate.

In particular:

- Fit feature scalers on the training split only.
- Use validation data for early stopping/model selection.
- Evaluate the final test set only after model selection is finished.
- Do not use final-test metrics as a training-time checkpoint criterion.
- Keep blackout windows chronologically or trajectory-separated according to the experiment definition.
- Do not introduce arbitrary time-lag corrections unless the synchronization analysis demonstrates a real lag.

This matters especially for the heading branch, where using the test loader every epoch would leak test information into model selection.

---

# 39. Training Sample Contract

For one 10-second example:

```text
Raw acceleration:
    [100, 3]

Raw gyro:
    [100, 3]
```

After 1-second grouping:

```text
Acceleration features:
    [10, 18]

Gyroscope features:
    [10, 18]

Vᵣ:
    [1]
```

### NCN

```text
Acceleration:
    18 → 32 → 64

Gyroscope:
    18 → 32 → 64

Concatenate:
    64 + 64 = 128

GRU:
    128 → hidden 128

Append Vᵣ:
    128 + 1 = 129

Regression:
    129 → 64 → 32 → 16 → 1

Output:
    N = [10, 1]
```

### MTN

```text
1-second acceleration preintegration:
    [100,3] → [10,3]

Gravity reference:
    [10,3]

Concatenate:
    [10,6]

Embedding:
    6 → 32 → 64

TCN:
    3 × (k=2, d=1)
    3 × (k=2, d=2)
    64 channels

Output:
    64 → 3

Angles:
    [10,3]
```

### Physics

```text
angles [10,3]
       ↓
rotation matrices [10,3,3]

raw acceleration [100,3]
       ↓
gravity removal
       ↓
phone → vehicle transform
       ↓
vehicle +Y projection
       ↓
1-second integration
       ↓
Δv_physics [10]

Δv_physics + N
       ↓
Δv [10]
       ↓
recursive speed
```

---

# 40. Approximate NCN Size

The architecture is intentionally small.

Approximate components:

```text
Acceleration embedding:
    18 → 32 → 64

Gyro embedding:
    18 → 32 → 64

GRU:
    input  = 128
    hidden = 128

Regression:
    129 → 64 → 32 → 16 → 1
```

The NCN is on the order of roughly `10^5` parameters, depending on the exact framework implementation and bias conventions.

Do not treat an approximate hand calculation as the authoritative parameter count. The actual implementation should report:

```python
sum(p.numel() for p in model.parameters())
```

---

# 41. Deployment Plan

The reference paper reports Android deployment with **ONNX Runtime**.

Cynos can use TFLite as a separate deployment choice.

A practical deployment split is:

## `velocity_dvse.tflite`

Preferred inputs:

```text
acc_features  [1,10,18]
gyro_features [1,10,18]
Vᵣ            [1]
I             [1,10,3]
```

where:

```text
I = one-second acceleration preintegration
```

Outputs:

```text
speed        [10]
delta_v      [10]
orientation  [10,3]
```

or the corresponding tensors needed by the native physics wrapper.

---

## `heading_tcn.tflite`

Input:

```text
gyro [1,100,3]
```

Output:

```text
yaw_rate [1,100,1]
```

Native application code then performs:

```text
rotation handling
heading integration
ΔE / ΔN
GNSS state management
Viterbi
NHC
```

This keeps the neural graphs focused on neural inference and the navigation logic outside the model.

---

# 42. Quantization Plan

Do not jump directly to INT8.

Use:

```text
FP32 training
     ↓
FP32 TFLite
     ↓
INT8 representative calibration
     ↓
INT8 evaluation
```

The representative calibration set should cover:

```text
normal driving
acceleration
braking
turning
stationary periods
different phone placements
different vehicle speeds
```

The required test is not merely whether the quantized model runs. Compare:

```text
velocity MAE
velocity P80
heading-rate error
heading-change error
position error at 2/5/10/15/30/60 s
latency
memory
```

---

# 43. Final Reference-Aligned Model Specification

## 43.1 Velocity Model

```text
============================================================
                       VELOCITY MODEL
============================================================

Input
    accel         [100, 3]
    gyro          [100, 3]
    Vᵣ            [1]

1-second feature extraction
    accel → [10,18]
    gyro  → [10,18]

NCN
    Acc:
        18 → 32 → 64

    Gyro:
        18 → 32 → 64

    concatenate:
        64 + 64 = 128

    GRU:
        input_size  = 128
        hidden_size = 128
        layers      = 1

    concatenate Vᵣ:
        128 + 1 = 129

    regression:
        129 → 64 → 32 → 16 → 1

    output:
        N [10,1]


MTN
    one-second acceleration preintegration:
        [100,3] → [10,3]

    concatenate gravity:
        [10,3] + [10,3] → [10,6]

    embedding:
        6 → 32 → 64

    TCN:
        Block 1:
            3 × (kernel=2, dilation=1)
            channels=64

        Block 2:
            3 × (kernel=2, dilation=2)
            channels=64

        receptive field = 10

    output:
        64 → 3

    angles:
        α β γ [10,3]

    rotation:
        Rᵥᵖ = Rₓ(α) Rᵧ(β) R_z(γ)


Physics
    raw acceleration
        ↓
    gravity removal
        ↓
    phone → vehicle transform
        ↓
    vehicle +Y projection
        ↓
    1-second integration
        ↓
    Δv_physics

    Δv = Δv_physics + N

    v[k] = v[k-1] + Δv
```

---

## 43.2 Heading Model

```text
============================================================
                        HEADING MODEL
============================================================

Input
    gyro [100,3]

TCN
    channels = 48
    dilations = [1,2,4,8]
    causal
    output channels = 1

Output
    yaw rate [100,1]

Integration
    yaw rate
       ↓
    Δψ
       ↓
    ψ
```

---

# 44. Clean Conceptual Split

The reference paper contributes the **velocity estimator**:

```math
\boxed{
\text{NCN}
+
\text{MTN}
+
\text{physics}
+
\text{integration}
}
```

Cynos adds the **navigation layer**:

```math
\boxed{
\text{GyroTCN}
+
\text{heading integration}
+
\text{GNSS handover}
+
\text{Viterbi/NHC}
}
```

The combined navigation state is:

```math
\boxed{
(\Delta E,\Delta N,v,\psi)
}
```

with GNSS position taking precedence whenever GNSS is valid.

---

# 45. Implementation Checklist

Before implementation is considered paper-aligned, verify:

### Velocity core
- [ ] 1-second statistical features
- [ ] 18 acceleration features
- [ ] 18 gyro features
- [ ] `Vᵣ` input
- [ ] NCN embeddings `18 → 32 → 64`
- [ ] one-layer GRU, hidden 128
- [ ] regression `129 → 64 → 32 → 16 → 1`
- [ ] one-second acceleration preintegration
- [ ] gravity reference `[0,0,9.81]`
- [ ] MTN TCN with 2 blocks × 3 causal layers
- [ ] kernel 2
- [ ] dilations 1 and 2 by block
- [ ] 64 channels
- [ ] 3 Euler-angle outputs
- [ ] `Rᵥᵖ = RₓRᵧR_z`
- [ ] explicit gravity removal
- [ ] vehicle `+Y` forward projection
- [ ] `Δv = Δv_physics + N`
- [ ] recursive velocity integration
- [ ] SmoothL1 objective
- [ ] `0.7 L_Δv + 0.3 L_v`

### Cynos extension
- [ ] gyro bias correction
- [ ] heading TCN
- [ ] yaw-rate target `-vehicle_yaw_rate`
- [ ] no second negation
- [ ] GNSS course initialization
- [ ] heading integration
- [ ] speed + heading → displacement
- [ ] GNSS/DR state machine
- [ ] Viterbi outside the model
- [ ] NHC outside the model

### Evaluation
- [ ] training/validation/test separation
- [ ] test not used for checkpoint selection
- [ ] train-only scaler fitting
- [ ] exact timestamp alignment for vehicle-ground-truth experiments
- [ ] blackout evaluation at 2/5/10/15/30/60 seconds

---

# 46. Bottom Line

The implementation should **not** be described as:

```text
features → GRU → TCN → velocity
```

That is the earlier Cynos baseline.

The paper-faithful velocity architecture is:

```text
                ┌────────── NCN GRU ────────── N
                │
IMU statistics ─┤
                │
                └────────── MTN TCN ───────── Rᵥᵖ
                                             │
raw acceleration ────────────────────────────┤
                                             ▼
                                      gravity removal
                                             ↓
                                      phone → vehicle
                                             ↓
                                      forward projection
                                             ↓
                                      physics integration
                                             ↓
                                         Δv_physics
                                             │
                                             + N
                                             ↓
                                            Δv
                                             ↓
                                            v
```

Then Cynos adds:

```text
corrected gyro
      ↓
GyroTCN
      ↓
yaw rate
      ↓
heading
      ↓
v + heading
      ↓
ΔE, ΔN
      ↓
dead-reckoned trajectory
```

This keeps the paper's **NCN + MTN + physics** structure intact while treating heading and road-constrained navigation as explicit Cynos extensions rather than pretending they are part of DVSE.
