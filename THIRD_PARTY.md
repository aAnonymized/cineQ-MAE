# Supplementary Material

## Implementation and Evaluation Protocols

### Contents

- [I. Disease Classification](#classification)
- [II. Volumetric Cardiac Segmentation](#segmentation)
- [III. Direct Biventricular EF Regression](#ef)
- [IV. DIS Pseudo-Supervision and Query Implementation](#dis)
- [V. CMAC Landmark Correspondence Evaluation](#cmac)
- [References](#references)

This supplement details the task-specific networks, objectives, optimization, validation, and model-selection procedures used for classification, segmentation, and ejection-fraction (EF) regression. It additionally specifies DIS-based pseudo-supervision and the CMAC landmark correspondence evaluation. Clinical cohort descriptions and disease-distribution tables are not repeated here. The task-specific protocols are summarized in Table [S1](#table-s1); their preprocessing and checkpoint-selection rules are not interchangeable.

<a id="classification"></a>

## I. Disease Classification

### A. Task definition and cine inputs

Two classifiers are trained independently: a two-class model for normal versus diseased subjects, and an eight-class model for disease subtyping after excluding normal subjects. These are separate supervised tasks, not a jointly optimized cascade: evaluation of the eight-class model is not conditioned on the binary model's predictions. Each task loads its fixed patient-level training and evaluation manifests. Repeated seeds use the same patients and class ordering.

Each patient is represented by three grayscale short-axis slice videos. The selected native slice indices are $\lfloor Z/2\rfloor+\{-2,0,2\}$, clipped to the available stack; short stacks can therefore contain repeated indices. The cache uses RAS-oriented indexing while retaining the identity of the acquired short-axis slice axis. Within each acquired plane, images are resampled to $0.994\times0.994$ mm; the native slice planes and through-plane spacing are retained. The fixed centered $224\times224$ field of view covers approximately $222.656\times222.656$ mm. Sampling outside the acquired plane is zero-padded. Joint first- and 99th-percentile intensity normalization over the selected videos is followed by clipping to $[0,1]$. Periodic linear interpolation produces 16 phases over the available cardiac cycle, starting at ED when its index is available and at frame zero otherwise. Single-frame inputs are excluded.

The resulting patient tensor has dimensions $3\times1\times16\times224\times224$ (slice, channel, time, height, width). Training and evaluation read the same completed image and trajectory caches. No optical flow or Query inference is performed inside the classification optimization loop.

### B. Encoder-only network and trajectory pooling

The pretrained encoder contains 12 transformer blocks with embedding dimension $D=768$ and 12 attention heads. Nonoverlapping $4\times16\times16$ tubelets yield a $4\times14\times14$ token grid. All tokens are visible during downstream learning. The reconstruction decoder, motion-query module, and mask token are not part of the classifier. A shared encoder processes the three slice videos, and all encoder parameters are fine-tuned with the classification head.

Let $\mathbf z_{v,h,p}\in\mathbb R^D$ be an encoder token for slice $v$, temporal token group $h$, and spatial position $p$. Cached displacement and local-deformation descriptors, $K_{v,h,p}$ and $R_{v,h,p}$, define the baseline-preserving pooling weight

<a id="eq-s1"></a>

$$
\begin{aligned}
w_{v,h,p}={}&1+0.3\frac{K_{v,h,p}}
 {\operatorname{mean}_{p'}K_{v,h,p'}+\epsilon}\\
&+0.7\frac{R_{v,h,p}}
 {\operatorname{mean}_{p'}R_{v,h,p'}+\epsilon},\qquad\epsilon=10^{-6}.
\end{aligned}
\tag{S1}
$$

Normalization is across spatial positions within each slice and source phase, not across patients. Writing the normalized descriptors as $\widetilde K$ and $\widetilde R$, the common downstream guidance rule is $w=1+0.3\widetilde K+0.7\widetilde R$. The unit baseline retains contributions from locations with zero motion descriptors; $w\geq1$ and is not bounded above. Weights are detached from autograd and remain fixed during fine-tuning. The phase-specific pooled feature is

<a id="eq-s2"></a>

$$
\mathbf z^w_{v,h}=
\frac{\sum_p w_{v,h,p}\mathbf z_{v,h,p}}
 {\max(\epsilon,\sum_p w_{v,h,p})}.
\tag{S2}
$$

If both descriptor maps are zero, every weight equals one and the operation reduces to ordinary spatial mean pooling.

For each slice, the four weighted phase features are concatenated with the unweighted global token mean and population standard deviation. The three slice descriptors are then concatenated, giving $3(4+2)D=13{,}824$ features. These global appearance summaries complement the baseline-preserving, trajectory-weighted features. The head comprises layer normalization, dropout with probability 0.7, and a linear layer with either two or eight outputs. The same trajectory-weighted pooling is used at training and inference.

### C. Objective and class-balanced sampling

For a task with $C$ output classes, the supervised objective is cross-entropy with label smoothing $\varepsilon_{\mathrm{LS}}=0.1$:

<a id="eq-s3"></a>

$$
\begin{aligned}
q_{ic}&=(1-\varepsilon_{\mathrm{LS}})\mathbf{1}[y_i=c]
                +\varepsilon_{\mathrm{LS}}/C,\\
\mathcal L_{\mathrm{cls}}&=-\frac1B\sum_{i=1}^{B}
                       \sum_{c=1}^{C}q_{ic}\log p_{ic}.
\end{aligned}
\tag{S3}
$$

There are no downstream reconstruction, DIS, or DPCC losses. Imbalance is addressed by sampling patients with replacement using weights proportional to $1/n_{y_i}$, where $n_c$ is the training count of task class $c$. Each epoch makes $N$ draws for $N$ training patients, giving each task class expected probability $1/C$. The binary sampler balances the two binary labels, not the constituent disease subtypes. No additional inverse-frequency class weights are applied in the loss. Evaluation uses each included patient once without balanced sampling.

### D. Optimization, validation, and model selection

AdamW is used with learning rates $10^{-5}$ for the encoder and $10^{-3}$ for the head, $(\beta_1,\beta_2)=(0.9,0.999)$, and weight decay 0.05. Training runs for 100 epochs with batch size eight, FP32 computation, and gradient-norm clipping at five. The schedule uses five warmup epochs followed by cosine decay to zero. Each minibatch produces one optimizer update; gradient accumulation is not used.

Augmentation applies a horizontal flip with probability 0.5, intensity scaling in $[0.9,1.1]$, and an additive brightness shift in $[-0.05,0.05]$, followed by clipping to $[0,1]$. Transformations are shared across a patient's slices and phases. The spatial weight map is flipped with the image. Temporal shifts are disabled because the cached weights are tied to fixed source phases. No augmentation is used at test time.

The cache-reuse classification protocol does not create an additional validation partition or select a validation-best checkpoint. It trains on the complete fixed task training list and evaluates the final 100th-epoch checkpoint, `last.pt`. There is no early stopping or test-driven checkpoint or decision-threshold selection in this loop. Seeds 0 and 1 are run separately. Seed-specific results and the equal average of their class-probability vectors are reported separately; the latter is a two-model ensemble, not an additional training run. Periodic checkpoints include optimizer, scheduler, and random-generator states to resume an interrupted run without restarting the schedule.

### E. Prediction and evaluation conventions

Predictions use the maximum softmax probability over the full task vocabulary. Probabilities are not renormalized over only the classes present in a particular evaluation cohort. For binary reevaluation, eligibility is determined from the harmonized diagnosis before mapping to normal/disease: only diagnoses represented in the actual private-source training records are retained. Other diagnoses are excluded rather than automatically counted as disease. Disease subtyping excludes normal subjects and diagnoses outside the fitted eight-class vocabulary. Saved probabilities can be filtered and rescored without retraining or changing the decision rule.

Accuracy, macro-F1, MCC, and macro-AUC are computed per evaluation cohort. The primary macro-F1 averages classes with nonzero reference support; a separate all-output-class macro-F1 is retained in the audit. One-vs-rest AUC is calculated only for classes with both positive and negative reference examples, and macro-AUC averages these valid AUCs. Undefined AUCs are not replaced by zero. Per-class recall is $\mathrm{TP}_c/(\mathrm{TP}_c+\mathrm{FN}_c)$; a class without reference support has undefined recall. With disease as the positive class, normal recall is specificity and disease recall is sensitivity. Membership-overlap flags are retained when forming evaluation summaries; an all-record score is distinguished from the score for records not selected for task training.

<a id="table-s1"></a>

**Table S1.** Task-specific architecture, optimization, and model-selection settings. Classification and EF use encoder-only prediction networks; segmentation uses a QCD-MAE V5 volumetric adapter. All three tasks use the guidance rule $w=1+0.3\widetilde K+0.7\widetilde R$.

| Item | Classification | Segmentation | Biventricular EF regression |
| --- | --- | --- | --- |
| Input unit | Three 16-phase slice videos | Separate ED or ES volume | Three 16-phase slice videos |
| Training input size | $3\times1\times16\times224^2$ | $1\times16\times160^2$ | $3\times1\times16\times224^2$ |
| In-plane spacing | $0.994\times0.994$ mm | $1.5\times1.5$ mm | $0.994\times0.994$ mm |
| Patch-embedding kernel | $4\times16\times16$ | $2\times16\times16$ | $4\times16\times16$ |
| Task head | LayerNorm-Dropout-Linear | Multiscale 3D adapter | LayerNorm-Dropout-Linear |
| Trajectory guidance | Normalized weighted pooling | Spatial feature modulation | Normalized weighted pooling |
| Outputs / head dropout | 2 or 8 / 0.7 | 4 voxel labels / not used | RVEF and LVEF / 0.2 |
| Loss | Smoothed cross-entropy | Cross-entropy + foreground Dice | Standardized dual-target MSE |
| Epochs / batch size | 100 / 8 patients | 1000 / 8 patches | 100 / 8 patients |
| Optimizer | AdamW | SGD, Nesterov momentum 0.99 | AdamW |
| Encoder / head learning rate | $10^{-5}$ / $10^{-3}$ | $10^{-2}$ / $10^{-2}$ | $10^{-5}$ / $10^{-3}$ |
| Weight decay | 0.05 | $3\times10^{-5}$ | 0.05 |
| Schedule | 5-epoch warmup + cosine | Polynomial, exponent 0.9 | 5-epoch warmup + cosine |
| Precision / gradient clipping | FP32 / norm 5 | Mixed precision / norm 12 | FP32 / norm 5 |
| Validation use | No validation-based selection | Every 20 epochs and final epoch | Monitored each epoch only |
| Evaluated parameters | Final epoch (100) | Highest validation mean Dice | Final epoch (100) |
| Random seeds | 0, 1 | 0, 1, 2 | 0 |

<a id="segmentation"></a>

## II. Volumetric Cardiac Segmentation

### A. ED/ES inputs and spatial preprocessing

ED and ES are processed as separate three-dimensional images with the same segmentation network, rather than concatenated into a two-frame sequence. Each volume is reoriented to a closest RAS representation and resampled onto an axis-aligned grid with spacing $(1.5,1.5,d_z)$ mm, where $d_z$ is the third voxel size after canonicalization. Images use linear interpolation and masks use nearest-neighbor interpolation. Clipped first-to-99th-percentile normalization maps intensities to $[0,1]$. Labels are harmonized to background, RV cavity, myocardium, and LV cavity. Unlike the classification cache, this axis-aligned resampling does not preserve the original oblique acquisition planes.

The complete processed volume is retained. Training samples $16\times160\times160$ depth-height-width patches rather than resizing an entire stack to 16 slices. Undersized dimensions are padded with zero image intensity and background labels. Foreground oversampling is 0.33. ED and ES from a patient always belong to the same partition.

### B. Encoder transfer and segmentation decoder

The pretrained patch-embedding convolution is retained, with its first axis interpreted as spatial depth rather than cardiac time. The V5 kernel and stride are $2\times16\times16$. The encoder has 12 blocks, dimension 768, and 12 attention heads. Positional embeddings are reshaped from an $8\times14\times14$ token grid and trilinearly interpolated to the patch grid $8\times10\times10$. This is an initialization for supervised spatial adaptation, not an assumption of physical equivalence between time and slice position. The entire encoder is fine-tuned; pretraining reconstruction and query heads are not used for segmentation.

Trajectory guidance acts on spatial encoder features rather than globally pooling them. For a feature channel $c$ and spatial token location $\mathbf r$, let the normalized descriptors be aligned with the feature grid. The guided feature is

$$
\widetilde F_c(\mathbf r)=
\bigl[1+0.3\widetilde K(\mathbf r)+0.7\widetilde R(\mathbf r)\bigr]
F_c(\mathbf r).
$$

The scalar weight is broadcast across channels, preserving the three-dimensional spatial layout for the segmentation decoder. Descriptor normalization follows [(S1)](#eq-s1), with no new clipping or change to the 0.3/0.7 coefficients. The guidance is fixed and detached; segmentation gradients update the encoder and decoder, not the trajectory estimator. Zero descriptors leave a feature unchanged because the weight is one. Spatially corresponding image patches and descriptor maps must share the same crop and geometric augmentation; slice depth is not reinterpreted as cardiac phase.

The custom `VolumeSegmenter` decoder uses transformer features from blocks 3, 6, 9, and 12. These are reshaped to three-dimensional grids and projected with $1\times1\times1$ convolutions to 32, 64, 128, and 256 channels. The first three are resized to one-half, one-quarter, and one-eighth of the input resolution. The deepest feature retains its token-grid resolution and is refined by an EPA3D block. Decoder blocks project and resize the incoming feature, concatenate the corresponding skip, and apply a $3\times3\times3$ convolution, instance normalization, GELU, and EPA3D refinement. EPA3D combines a depthwise spatial-convolution branch and a global-average-pooled channel-gating branch through a pointwise projection and residual connection. A shallow 32-channel image branch supplies the full-resolution skip. The final $1\times1\times1$ convolution produces four-class logits. Auxiliary heads are attached to the half-resolution decoder feature and the quarter-resolution projected encoder feature.

### C. Segmentation loss and augmentation

Each supervised output uses cross-entropy plus a foreground soft Dice loss. For foreground class $c$, let $p_{ic}$ and $g_{ic}$ be its probability and one-hot reference; sums are over valid batch voxels:

<a id="eq-s4"></a>

$$
\mathcal L_{\mathrm{Dice}}=1-\frac13\sum_{c=1}^{3}
\frac{2\sum_i p_{ic}g_{ic}+\epsilon}
 {\sum_i p_{ic}+\sum_i g_{ic}+\epsilon},\quad\epsilon=10^{-5}.
\tag{S4}
$$

For outputs $l=0,1,2$, the deep-supervision objective is

<a id="eq-s5"></a>

$$
\mathcal L_{\mathrm{seg}}=\sum_{l=0}^{2}
 \frac{2^{-l}}{1.75}
 (\mathcal L_{\mathrm{CE}}^{(l)}+\mathcal L_{\mathrm{Dice}}^{(l)}).
\tag{S5}
$$

Masks are resized to each output with nearest-neighbor interpolation. Outside-volume labels introduced by spatial augmentation are set to $-1$ and ignored by the supervised losses. Augmentation settings are listed in Table [S2](#table-s2). Elastic deformation is disabled.

<a id="table-s2"></a>

**Table S2.** Volumetric segmentation augmentation. Image spatial interpolation is cubic; masks use a label-aware spatial transform. Unspecified noise and contrast ranges retain the installed batchgenerators defaults.

| Transform | Range and probability |
| --- | --- |
| Rotation / scale | $\pm30^\circ$ per axis / $[0.7,1.4]$; $p=0.2$ each |
| Gaussian noise | $p=0.1$ |
| Gaussian blur | $\sigma\in[0.5,1]$; sample $p=0.2$, channel $p=0.5$ |
| Brightness | Multiplicative $[0.75,1.25]$; $p=0.15$ |
| Contrast | $p=0.15$ |
| Low-resolution simulation | Zoom $[0.5,1]$; sample $p=0.25$, channel $p=0.5$ |
| Gamma | $[0.7,1.5]$; inverted $p=0.1$, noninverted $p=0.3$; retain statistics |
| Mirroring / clipping | All three spatial axes / final $[0,1]$ clipping |

### D. Optimization, validation, and model selection

The V5 protocol uses SGD with Nesterov momentum 0.99, initial learning rate 0.01, and weight decay $3\times10^{-5}$. Encoder and decoder have the same learning rate. Training runs for 1000 epochs with 250 minibatches per epoch, batch size eight, mixed precision, gradient-norm clipping at 12, and no gradient accumulation. Learning rates follow polynomial decay with exponent 0.9 and no warmup.

For ACDC, the validation partition contains 20% of each class from the official training partition, using subset seed 2026. For M&Ms-1/2, the official training, validation, and test assignments are retained, restricted to subjects with reference masks. Reduced-label experiments subsample only the training partition; configured training fractions are 20%, 50%, and 100%, with random seeds 0, 1, and 2. Validation runs every 20 epochs and at the final epoch. The checkpoint with the highest validation mean foreground Dice is selected; a tie retains the earlier checkpoint. The V5 configuration also monitors test performance after validation, but those test scores do not enter the programmatic checkpoint selector. Final results are evaluated from the validation-selected checkpoint, not automatically from epoch 1000.

### E. Sliding-window inference and native-space metrics

Inference covers the full processed volume using sliding windows with step size equal to half the patch extent and inference batch size one. Overlapping class probabilities are blended by a Gaussian importance map with standard deviation one-eighth of the patch extent along each axis. Temporary padding is removed. Each class probability volume is linearly resampled through the stored affine to the corresponding original ED or ES image grid. The discrete prediction is obtained by argmax after this restoration; reference masks are aligned to the same native grid using nearest-neighbor interpolation.

Dice and HD95 are computed separately for RV, myocardium, and LV. HD95 is the 95th percentile of the concatenated bidirectional surface distances, calculated using native voxel spacing and reported in millimeters. If both masks are empty, the structure's Dice and HD95 are undefined and excluded from finite-value averages. If only one is empty, Dice is zero and HD95 is the physical image-grid diagonal, $\|(\mathbf n-\mathbf1)\odot\mathbf d\|_2$. Scores are averaged over ED/ES within patient before averaging across patients. Valid contributor counts are retained when empty structures affect aggregation.

<a id="ef"></a>

## III. Direct Biventricular EF Regression

### A. Reference values and input coverage

Reference RVEF and LVEF are computed from the original complete ED/ES cavity segmentations, not predicted masks or the three input slices. Following the volumetric EF convention used in CineMA [[1]](#ref-1), for ventricle $c\in\{\mathrm{RV},\mathrm{LV}\}$,

<a id="eq-s6"></a>

$$
\begin{aligned}
V_c^q&=N_c^q d_xd_yd_z/1000\quad \mathrm{mL},\quad q\in\{\mathrm{ED},\mathrm{ES}\},\\
y_c&=100(V_c^{\mathrm{ED}}-V_c^{\mathrm{ES}})/V_c^{\mathrm{ED}}.
\end{aligned}
\tag{S6}
$$

Voxel sizes are converted to millimeters using NIfTI unit metadata; when units are unspecified, the configured millimeter assumption is recorded in the reference audit. Native cavity labels are RV/LV=1/3 for ACDC and 3/1 for M&Ms-1/2. Four-dimensional masks require explicit ED/ES indices. Both chambers must satisfy $0<V^{\mathrm{ES}}\leq V^{\mathrm{ED}}$.

Patients with missing or invalid references, unusable cine arrays, or missing trajectory descriptors are excluded with their reasons recorded. Exclusion is patient-wide and common to both targets and all methods within the dataset; ED/ES are not interchanged to repair a negative EF. Duplicate identifiers and cross-partition conflicts stop the workflow. Common quality control precedes the fixed split. The model reads the same three-slice, 16-phase, $0.994$-mm cine cache as classification (Section [I](#classification)). Thus, the input covers the cardiac cycle but not the complete ventricular stack. Reference masks are used only to derive targets, never as model inputs.

### B. Encoder, regression head, and loss

The encoder and feature concatenation are those in [(S1)](#eq-s1)-[(S2)](#eq-s2), including frozen trajectory weights $w=1+0.3\widetilde K+0.7\widetilde R$ and the unweighted global mean/standard-deviation features. The descriptor normalization and unit baseline are identical to classification. A new head applies layer normalization, dropout 0.2, and a linear mapping from 13,824 features to two outputs ordered as RVEF and LVEF. No reconstruction decoder or segmentation network participates in EF prediction. A single model jointly predicts the two EF values.

Each target is standardized using only the current dataset's training patients. With training mean $\mu_c$ and population standard deviation $\sigma_c$, the objective is

<a id="eq-s7"></a>

$$
\mathcal L_{\mathrm{EF}}=\frac{1}{2B}\sum_{i=1}^{B}
 \sum_{c\in\{\mathrm{RV},\mathrm{LV}\}}
 \left(\hat z_{ic}-\frac{y_{ic}-\mu_c}{\sigma_c}\right)^2.
\tag{S7}
$$

Predictions are transformed back as $\hat y_{ic}=\sigma_c\hat z_{ic}+\mu_c$ without post-hoc clipping. The output is direct image-to-EF regression, not an EF calculation from predicted segmentations.

### C. Independent training, validation, and selection

ACDC, M&Ms-1, and M&Ms-2 are trained and tested separately. Each dataset-method pair starts a new regressor from its corresponding pretrained encoder. VideoMAE, HeartMAE, TrackMAE, and CardiacQ-MAE use the same valid patients, head, optimization, and frozen Query-derived pooling weights. This compares encoder initializations under a common downstream pipeline; it does not reproduce every baseline's original downstream architecture.

For ACDC, two patients per diagnostic class are reserved for validation from the official training partition using split seed 42. M&Ms-1/2 retain their official partitions. All diagnoses with valid biventricular references, including normal subjects, are eligible. Subsets and target standardization are shared across the four methods within each dataset. Training uses ordinary shuffled patient minibatches, not classification's inverse-class-frequency sampler.

All runs use seed 0 and 100 epochs. AdamW settings, batch size, FP32 precision, gradient clipping, warmup/cosine schedule, and synchronized augmentation follow the classification settings in Table [S1](#table-s1), except for the regression-specific loss and dropout 0.2. Validation RVEF, LVEF, and mean MAE are monitored after each epoch but do not select a checkpoint or trigger early stopping. The final epoch's parameters are evaluated on the test partition once. In the memory-only implementation, "last" means these final in-memory parameters; no validation-best parameters are restored.

### D. Error statistics and reproducible plots

For patient $i$, the three absolute-error entries are

<a id="eq-s8"></a>

$$
\begin{aligned}
a_i^{\mathrm{RV}}&=|\hat y_i^{\mathrm{RV}}-y_i^{\mathrm{RV}}|,\qquad
a_i^{\mathrm{LV}}=|\hat y_i^{\mathrm{LV}}-y_i^{\mathrm{LV}}|,\\
a_i^{\mathrm{Mean}}&=(a_i^{\mathrm{RV}}+a_i^{\mathrm{LV}})/2.
\end{aligned}
\tag{S8}
$$

The Mean entry averages the two absolute errors within a patient; it is not the absolute error of the averaged EF values. For each dataset-method-entry combination, the reported statistics are

<a id="eq-s9"></a>

$$
\begin{aligned}
\mathrm{MAE}&=n^{-1}\sum_i a_i,\\
s&=\sqrt{\frac{\sum_i(a_i-\mathrm{MAE})^2}{n-1}},\qquad
\mathrm{Var}=s^2.
\end{aligned}
\tag{S9}
$$

MAE and standard deviation are in percentage points (pp), and variance is in pp$^2$. The summary additionally reports the sample count, minimum, maximum, P05, Q1, median, Q3, P95, and mean $\pm$ SD. Boxes span Q1-Q3 with the median as their center line; whiskers use P05-P95. The printed values therefore reconstruct the plotted boxes without individual predictions. With one training seed, SD describes dispersion across test patients, not between-run variation, a confidence interval for MAE, or scan-rescan repeatability. No scan-rescan coefficient of variation is estimated.

<a id="dis"></a>

## IV. DIS Pseudo-Supervision and Query Implementation

### A. Offline optical flow and forward-backward confidence

Offline optical flow uses Dense Inverse Search (DIS) [[2]](#ref-2). The OpenCV implementation is initialized through `DISOpticalFlow_create` with its `MEDIUM` preset [[3]](#ref-3). Normalized frames are bilinearly resized from $224^2$ to $112^2$, clipped to $[0,1]$, multiplied by 255, rounded, and converted to unsigned 8-bit integers. Forward and backward flows are estimated independently from each source $s\in\{0,4,8,12\}$ to every $t\ne s$, without an initial flow estimate. These are direct source-to-target flows, not compositions of adjacent-frame flow. One OpenCV CPU thread is used; internal search/refinement parameters are inherited from the selected library preset.

Let $\eta=112/224$. A model-image source coordinate $\mathbf a$ maps to the flow grid as $\mathbf a_f=\eta(\mathbf a+1/2)-1/2$. With forward and backward fields $\mathbf f_{s\to t}$ and $\mathbf f_{t\to s}$, define

<a id="eq-s10"></a>

$$
\mathbf b_f=\mathbf a_f+\mathbf f_{s\to t}(\mathbf a_f),
\tag{S10}
$$

<a id="eq-s11"></a>

$$
\mathbf e_f=\mathbf f_{s\to t}(\mathbf a_f)
                   +\mathbf f_{t\to s}(\mathbf b_f),
\tag{S11}
$$

<a id="eq-s12"></a>

$$
\mathbf y^{\mathrm{DIS}}=(\mathbf b_f+1/2)/\eta-1/2.
\tag{S12}
$$

Flow sampling is bilinear with replicated borders. The confidence is

<a id="eq-s13"></a>

$$
c_{s,t}(\mathbf a)=\mathbf{1}[\mathbf y^{\mathrm{DIS}}\in[0,223]^2]
 \exp\left(-\frac{\|\mathbf e_f\|_2}{\eta\sigma_{\mathrm{FB}}}\right),
\tag{S13}
$$

where $\sigma_{\mathrm{FB}}=2$ model-image pixels, equivalent to one flow-grid pixel. This is an exponential of the Euclidean consistency residual, not a Gaussian of its square. Confidence lies in $[0,1]$ and is not a calibrated probability of anatomical correctness. Out-of-image endpoints have zero confidence even though border replication permits numerical sampling. For $s=t$, identity coordinates and confidence one are assigned without computing optical flow.

The fixed pointwise reliability filter is

<a id="eq-s14"></a>

$$
\bar c_{j,r,t}=c_{j,r,t}\mathbf{1}[c_{j,r,t}\geq\delta],\qquad\delta=0.1.
\tag{S14}
$$

Equality is retained. For an in-bounds endpoint, this is equivalent to a forward-backward residual no larger than approximately 4.605 model-image pixels. The same threshold is used for centers and neighbors without patient-specific tuning.

### B. Reliable-query selection and coordinate loss

Candidate queries are masked tubelets whose mean center confidence over non-source phases is at least $\delta$. The budget is $\min(\lceil0.1N_{\mathrm{masked}}\rceil,N_{\mathrm{reliable}})$. The highest-motion candidates, ranked by cycle-averaged DIS displacement, provide $\lceil0.7N_{\mathrm{query}}\rceil$ queries; remaining queries are sampled without replacement from other reliable candidates. Insufficient reliable candidates reduce the budget rather than being replaced by unreliable points.

The coordinate penalty is

<a id="eq-s15"></a>

$$
\ell_{\mathrm{coord}}(\hat{\mathbf y},\mathbf y)=
\frac{1}{2p}\sum_{d\in\{x,y\}}
 \operatorname{SmoothL1}_{\beta=1}(\hat y_d-y_d).
\tag{S15}
$$

Here $p=16$. Smooth-$L_1$ is applied in pixels and divided by patch size afterward. The center, sampled-neighbor, and composed-endpoint terms have coefficients $1/4$, $1/4$, and $1/2$. A per-cine weighted term uses $\sum_i\omega_i\ell_i/\max(1,\sum_i\omega_i)$ before minibatch averaging. Local Jacobian matching uses Smooth-$L_1$ with transition parameter 0.1 and coefficient 0.1 within the DIS loss. For a composed path via phase $m$, endpoint reliability is $\bar c_{j,0,m}\bar c_{j,0,t}$, both measured relative to the original source anchor. It is not a separate confidence estimate for the second hop. Pseudo-coordinates and confidences are supervision data, not inputs to the learned coordinate query.

### C. Cyclic phase encoding and numerical ranges

For $T=16$, normalized time $u=t/T\in[0,1)$ is encoded using

<a id="eq-s16"></a>

$$
\begin{aligned}
\theta(u)&=\begin{cases}
3\pi u,&0\leq u\leq1/3,\\
\pi+\tfrac32\pi(u-1/3),&1/3<u<1,
\end{cases}\\
\boldsymbol\phi(t)&=[\sin\theta,\cos\theta,\sin2\theta,\cos2\theta].
\end{aligned}
\tag{S16}
$$

There are two temporal harmonics, giving four features for each source and target phase, all in $[-1,1]$. The fixed breakpoint $1/3$ is a modeling choice, not a detected ES phase. The uniform-phase ablation uses $\theta=2\pi u$. Complementary intermediate-phase sets are $\{0,\ldots,5\}$ and $\{6,\ldots,15\}$; the source and target phases are removed before sampling one intermediate phase from each set.

Dividing spatial coordinates by 223 maps in-bounds query inputs to $[0,1]^2$. Four spatial Fourier frequencies $\pi\{1,2,4,8\}$ are used per coordinate. Two normalized coordinates, 16 spatial sine/cosine features, and eight source/target phase features give a 26-dimensional query-embedding input. These four spatial frequencies must not be confused with the two temporal harmonics. The motion head produces bounded source-relative displacements, limited to 48 pixels per component and further bounded by the corresponding image border. Same-phase queries return the source coordinate exactly.

### D. Local reliability and empty neighborhoods

Four neighboring source points lie four pixels to the left, right, above, and below each center. Local deformation is evaluated at $\mathcal S_j=\{(s_j+o)\bmod16:o\in\{2,6,10,14\}\}$ using

<a id="eq-s17"></a>

$$
\mathbf J_{j,t}=\left[
\frac{\hat{\mathbf y}_{j,2,t}-\hat{\mathbf y}_{j,1,t}}{8},
\frac{\hat{\mathbf y}_{j,4,t}-\hat{\mathbf y}_{j,3,t}}{8}\right].
\tag{S17}
$$

The displacement descriptor averages the center displacement over phases with reliable centers. The deformation descriptor averages $\|\mathbf J_{j,t}-\mathbf I_2\|_F$ over phases in $\mathcal S_j$ for which the center and all four neighbors pass [(S14)](#eq-s14). An unreliable neighborhood contributes neither a deformation estimate nor Jacobian pseudo-supervision.

If either the reliable-center set or reliable-neighborhood set is empty, TGWR is not applied to that tubelet: its ordinary reconstruction weight remains one. Missing evidence is not treated as zero motion or replaced by an identity-Jacobian pseudo-label. Reliable center coordinate supervision can still contribute independently. If a cine has no reliable queries, its DIS and DPCC losses are zero while masked reconstruction remains active.

### E. Pretraining weights versus downstream guidance

The implemented pretraining normalization uses fixed-scale clipping:

<a id="eq-s18"></a>

$$
\begin{aligned}
\bar K_j&=\operatorname{clip}(K_j/1.2,0,1),\\
\bar R_j&=\operatorname{clip}(R_j/0.07,0,1),\\
\mu_j&=(\bar K_j^2+\bar R_j^2)/2.
\end{aligned}
\tag{S18}
$$

Here $K_j$ is measured in model pixels and $R_j$ is dimensionless; $\bar K_j$, $\bar R_j$, and $\mu_j$ all lie in $[0,1]$. This is neither min-max nor mean normalization. At zero-based epoch $e$, a reliably queried tubelet has reconstruction weight

<a id="eq-s19"></a>

$$
w_j^{(e)}=\operatorname{sg}\left[\min\left(1,\frac{e+1}{50}\right)\mu_j\right].
\tag{S19}
$$

There is no additive baseline for these reliable queries. Unqueried or unreliable tubelets retain unit weight. The weights are detached, so reconstruction gradients do not propagate through their computation. The reconstruction loss is normalized by the number of masked tubelets, not the sum of their weights. This pretraining definition is distinct from downstream feature guidance: classification, segmentation, and EF use $1+0.3\widetilde K+0.7\widetilde R$ with mean-normalized descriptors. Their unit baseline is retained even when both descriptors are zero. Changing the downstream rule does not redefine the pretraining reconstruction loss.

<a id="table-s3"></a>

**Table S3.** Key implementation constants for DIS supervision and query encoding.

| Item | Value |
| --- | --- |
| Model / flow image | $224^2$ / $112^2$ |
| DIS preset | OpenCV `MEDIUM` |
| Forward-backward scale | 2 model pixels = 1 flow-grid pixel |
| Confidence threshold | $\delta=0.1$, inclusive |
| Query fraction / high-motion share | 0.10 / 0.70 |
| Phases / source indices | 16 / $\{0,4,8,12\}$ |
| Temporal harmonics / warp split | 2 / $1/3$ |
| Spatial Fourier frequencies | $\pi\{1,2,4,8\}$ per axis |
| Coordinate / sinusoid range | $[0,1]$ / $[-1,1]$ |
| Neighbor offset / difference span | 4 / 8 model pixels |
| TGWR scales / ramp | 1.2 pixels, 0.07 / 50 epochs |
| Empty reliable neighborhood | Unit reconstruction weight; no invalid Jacobian target |

<a id="cmac"></a>

## V. CMAC Landmark Correspondence Evaluation

### A. Benchmark and annotation inclusion

CMAC provides multimodal cardiac-motion data from a dynamic phantom and 15 healthy volunteers, with 12 myocardial landmarks at four walls and three ventricular levels tracked by two observers in tagged MR images [[4]](#ref-4). Our evaluation uses the 15 human cine SSFP cases and the corresponding released reference landmarks; the phantom is excluded. Subject identifiers are $\{1,2,4,5,6,7,8,9,10,11,12,13,14,15,16\}$. There are 180 anatomical landmark sites and 360 subject-landmark-observer records before exclusions. Applying the release-specific exclusion list in Table [S4](#table-s4) leaves 352 records. Both observers are retained, but not counted as independent subjects. All methods are evaluated on the same included records.

<a id="table-s4"></a>

**Table S4.** CMAC reference exclusions. Landmark numbers are one-based; all eight exclusions concern observer 2.

| Subject | Observer | Landmark number(s) |
| --- | --- | --- |
| 7 | 2 | 2 |
| 10 | 2 | 6, 10 |
| 11 | 2 | 6 |
| 12 | 2 | 6, 9 |
| 13 | 2 | 6 |
| 14 | 2 | 11 |

<a id="table-s5"></a>

**Table S5.** Zero-based CMAC cine ES and tagged-reference ES indices. Each index belongs to its own sequence.

| Subject | 1 | 2 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cine ES | 10 | 11 | 11 | 11 | 11 | 9 | 9 | 10 | 10 | 9 | 11 | 9 | 10 | 11 | 11 |
| Reference ES | 10 | 10 | 10 | 10 | 10 | 11 | 10 | 10 | 10 | 11 | 10 | 10 | 11 | 8 | 9 |

### B. Frozen-model protocol and physical projection

No network is trained, fine-tuned, or selected using CMAC landmarks. Every included ED landmark is queried; the sparse pretraining query budget is not applied. The model receives the image sequence and ED source coordinate, but not the annotated ES endpoint. ES landmarks serve exclusively as evaluation references. EPE evaluates predicted motion endpoints and is distinct from appearance reconstruction at manually supplied target locations.

DICOM images are ordered by slice position along the slice normal and checked for consistent geometry over time. Let $\mathbf o_z$ be a slice's image-position origin, $\mathbf e_x$ and $\mathbf e_y$ its directions for increasing image column and row, and $s_x,s_y$ the respective pixel spacings. The SSFP VTK references in this release use image-axis millimeter coordinates. They are converted to DICOM LPS as

<a id="eq-s20"></a>

$$
\mathbf p^{\mathrm{LPS}}=\mathbf o_0+
[\mathbf e_x\ \mathbf e_y\ \mathbf e_z]
(\mathbf p^{\mathrm{VTK}}-\mathbf o_0),
\tag{S20}
$$

where $\mathbf o_0$ is the first-slice origin and $\mathbf e_z$ is the stack direction. This release-specific convention must not be applied to a VTK file already expressed in world coordinates.

The ED source landmark alone selects the nearest acquired plane:

<a id="eq-s21"></a>

$$
z^\star=\arg\min_z\left|
(\mathbf p_{\mathrm{ED}}^{\mathrm{LPS}}-\mathbf o_z)^\top
(\mathbf e_x\times\mathbf e_y)\right|.
\tag{S21}
$$

The source must be in the slice field of view and within the permitted half-slice-spacing tolerance of the stack. ED and ES references are projected onto that same fixed plane using

<a id="eq-s22"></a>

$$
x=\frac{(\mathbf p^{\mathrm{LPS}}-\mathbf o_{z^\star})^\top\mathbf e_x}{s_x},
\quad
y=\frac{(\mathbf p^{\mathrm{LPS}}-\mathbf o_{z^\star})^\top\mathbf e_y}{s_y}.
\tag{S22}
$$

The ES reference does not select a more favorable plane. Through-plane displacement is recorded in the geometry audit but not included in the reported in-plane endpoint distance.

### C. Model-grid and temporal mapping

The entire selected slice is resized to fit $224\times224$ pixels while preserving physical aspect ratio and is centrally padded. For native dimensions $H\times W$, let $g=224/\max(Hs_y,Ws_x)$, $H'=\operatorname{round}(Hs_yg)$, and $W'=\operatorname{round}(Ws_xg)$. With $a_x=W'/W$, $a_y=H'/H$, and padding $(b_x,b_y)$,

<a id="eq-s23"></a>

$$
x'=(x+\tfrac12)a_x-\tfrac12+b_x,\quad
y'=(y+\tfrac12)a_y-\tfrac12+b_y.
\tag{S23}
$$

The native 30-phase cine is normalized using joint first-to-99th percentiles and uniformly resampled to 16 phases by periodic linear interpolation. ED is time zero. The target query time is $16t_{\mathrm{ES}}^{\mathrm{cine}}/30$, which can be fractional. Cine ES and tagged-reference ES indices are stored separately (Table [S5](#table-s5)); a fixed ES fraction is not assumed.

### D. Correspondence metrics and statistical units

For model-grid prediction $(\hat x',\hat y')$ and projected reference $(x'_{\mathrm{ref}},y'_{\mathrm{ref}})$, physical endpoint error is

<a id="eq-s24"></a>

$$
\mathrm{EPE}=\sqrt{
\left[\frac{s_x}{a_x}(\hat x'-x'_{\mathrm{ref}})\right]^2+
\left[\frac{s_y}{a_y}(\hat y'-y'_{\mathrm{ref}})\right]^2}.
\tag{S24}
$$

Distances are in millimeters using subject- and axis-specific scale factors. The classification-cache spacing of 0.994 mm is not used as a universal conversion factor. PCK@2 mm and PCK@5 mm are the fractions of records with EPE no greater than the respective thresholds.

Errors and PCK indicators are first averaged over retained landmark-observer records within each subject. The reported point estimate is the equally weighted mean over the 15 subjects. The 95% confidence interval uses 2000 subject-level bootstrap resamples with seed 0 and percentile limits 2.5% and 97.5%. Paired method differences resample the same subjects jointly. The 352 reference records are not treated as 352 independent subjects. These metrics quantify projected in-plane correspondence, not full three-dimensional tracking or strain accuracy. Temporal and through-plane differences between cine and reference acquisitions remain limitations of this evaluation.

<a id="references"></a>

## References

<a id="ref-1"></a>

[1] Y. Fu *et al.*, "Development and validation of a versatile foundation model for cine cardiac magnetic resonance image analysis," *Commun. Med.*, vol. 6, Art. no. 408, 2026, doi: [10.1038/s43856-026-01636-0](https://doi.org/10.1038/s43856-026-01636-0).

<a id="ref-2"></a>

[2] T. Kroeger, R. Timofte, D. Dai, and L. Van Gool, "Fast optical flow using dense inverse search," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2016, pp. 471-488, doi: [10.1007/978-3-319-46493-0_29](https://doi.org/10.1007/978-3-319-46493-0_29).

<a id="ref-3"></a>

[3] OpenCV, "DISOpticalFlow class reference." Accessed: Oct. 21, 2026. [Online]. Available: [https://docs.opencv.org/4.x/de/d4f/classcv_1_1DISOpticalFlow.html](https://docs.opencv.org/4.x/de/d4f/classcv_1_1DISOpticalFlow.html)

<a id="ref-4"></a>

[4] C. Tobon-Gomez *et al.*, "Benchmarking framework for myocardial tracking and deformation algorithms: An open access database," *Med. Image Anal.*, vol. 17, no. 6, pp. 632-648, 2013, doi: [10.1016/j.media.2013.03.008](https://doi.org/10.1016/j.media.2013.03.008).
