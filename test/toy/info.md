processing from existing capture output then fallback to rerun capture to include input activations for selected layers. But maybe easier to use existing activation files as model inputs to per-layer emulation for Conv/Linear: For each layer in manifest, stored output of that layer, not input. Could still analyze "weight reconstruction error" independent of activation. But user asked use real activations data for error analysis. Could still use output activations to compute error through local Jacobian? Maybe not.

Need inspect test/capture_all_activations maybe we can modify script to capture inputs; but user requested new script; could script parse existing activation output and also run a forward pass? maybe loads model and from same inputs obtains activations at target layers (inputs) with hooks. But to use existing activation dataset maybe we can use manifest to choose layers and potentially to sample shapes.

Given user asked create script. We can create robust script under TinySR/test perhaps `analyze_layer_compression.py`. It can:

- Load model as in capture script with LoRA.
- Load manifest and run_meta from activation capture dir.
- Build candidate layers: modules with weight dims either 2D linear or Conv2d with kernel_size==1 and groups==1 maybe also optionally general Conv2d by flatten.
- For each candidate layer, gather real activation input from captured outputs via mapping? could not.
Alternative we can directly load activation tensors from manifest for that exact layer (which are outputs). But for error analysis of layer output, we need input. Could instead use captured output from layer as X for projecting through compressed weight transposed? not meaningful but maybe for row-space error? maybe we can compute `Y = X @ W.T` with X from output by requiring output dim equals in_dim maybe not.
Better approach: Use forward pre-hooks now to capture inputs using same input image from run_meta args; run model once. That's still real activations derived from real image. Could mention if existing capture has input data we can use else rerun ephemeral capture. But user specified path has activations. Could script optionally use existing manifest outputs for layer selection and for shape check, but compute inputs by replay.

Maybe do this:

- Script arguments:
  - `--activation_capture_dir` pointing to outputs/activations_capture_tinysr_env.
  - `--use_existing_activation_only` maybe false default.
- It loads run_meta args and image path from capture dir.
- Loads TinySR models with those args.
- Select candidate layers based on module type and also presence in manifest layer names (to ensure they were active).
- Register forward_pre_hook for candidate layers to capture first call input activations during a replay run using same input image and pooled embeddings etc.
- During replay, only record up to max tokens sample maybe maybe huge memory. Input activations huge; we can sample random spatial positions to reduce memory.
- For each layer call, maintain maybe one sample tensor (downsample) to avoid oom.
- Then run per-layer compression:

For Linear:
- weight shape [out,in]
- input activation shape maybe [B,N,in] or [B,in]
- flatten to (-1, in).
- Optionally subsample max_tokens e.g., 16384.
- calculate baseline output Y_orig = X @ W.T (+ bias? maybe ignore bias; but layer output includes bias. For error due weight only we can ignore bias since cancels when comparing both with same bias maybe if bias present and we include it in both). compute Yq with W_approx.
- relative error.

For Conv2d 1x1:
- weight shape [out,in,1,1]
- input activation [B,in,H,W]
- flatten X = input.permute(0,2,3,1).reshape(-1,in), W2d = weight[:,:,0,0].
- output flatten Y = X @ W2d.T.
- if bias apply add same.
- error.

Could also include optional Conv2d other kernels by unfolding:
- For kernel_size >1 and groups==1 maybe use F.unfold on sampled patches to matrix multiply.
But we planned suitable layers maybe linear + pointwise conv. so skip others.

Need include H-SVD and FFT-MixedQ4 compression with budgets ratios 0.2-0.7.
Need adapt FFT MixedQ4 if needed:
- 2D FFT on W maybe for small dimension may not effective.
- Could add fallback for very small matrices or extremely rectangular? maybe maintain.
- Could adjust for Conv1x1 by using hadamard pre-processing? maybe not.
Need "必要时对FFT-MixedQ4 方法做调整." We can include adjustment to handle 1x1 conv and robust budget:
- Additional option `--fft_axis_strategy`? maybe not.
- More crucial: enforce conjugate symmetry for real-valued weights? Actually weight real => fft complex with conjugate symmetry. If arbitrary top-k mask breaks symmetry, ifft yields complex then taking real discards imag causing inconsistency. Original function uses `.real`. Could adjust to symmetric mask to keep ifft result real and potentially better. This is likely necessary.

Need implement improved FFT-MixedQ4 maybe for real matrix use rfft2 to store only non-redundant half-spectrum and automatically real ifft2 from rfft. This would avoid symmetry issue and better budget mapping may change. Should we implement? Could be complicated but doable.

Given request maybe "if needed adjust method"; we can adjust in new script not touching old maybe implement `fourier_mixedq4_rfft`:
- Use `torch.fft.rfft2(W)` returns shape [m, n_r] complex where n_r=n//2+1.
- Each coefficient except DC/Nyquist maybe has conjugate partner implicitly so cost maybe? but for storage actual number of complex coefficients in rfft representation is m*n_r.
- For real output, if we manipulate rfft coefficients arbitrarily, irfft2 yields real exactly.
- Budget mapping now against original real params (m*n). In old function, full complex cost 2 per coefficient on full spectrum m*n leads 2*m*n real values, bigger than original; this budget inconsistent maybe but still.
For our layer compression we need W_approx under target param count maybe ratio of original params m*n.
Could design simpler:

- For target_params = int(ratio*W.numel()).
- Estimate keep_fp count K in rfft domain and quantize rest.
- Cost model in "FP32 real units relative to original real params".
In rfft representation each complex coefficient as 2 real FP32 (except some special purely real frequencies but ignore).
Then total full cost approx 2*Nf where Nf=m*n_r. This may exceed original m*n by about factor ~1+2/n? Actually n_r ~ n/2+1 so 2*Nf ~ m*(n+2) roughly slightly above original by +2m.
Could still use ratio budgets maybe impossible for high ratios? We can scale maybe treat budget base as full_rfft_cost. But user asked compare budgets maybe 20-70% of original params from hsvd_validation. Could maintain by cost normalization fudge.

Alternative easier: keep old fft2 full spectrum approach but adjust for real symmetry by symmetrizing filtered spectrum after quantization. But implementing symmetric mask for arbitrary top-k complex pairs maybe complex but manageable.

Given matrix W real; full fft2 shape m,n with conjugate: F[u,v] conj pair at [-u,-v]. To maintain real ifft, need F' obey symmetry. We can enforce by `F_sym = 0.5*(F' + conj(flip(flip(F',0),1)))` with proper periodic indexing not simple flip maybe shift due indices.
Could use inverse transform result real anyway by taking real part. But better maybe use rfft2.

Since script for analysis only, maybe we can keep old method but modify "for small matrix or conv1x1 we optionally center normalize". Hmm.

Need inspect candidate layers likely large linear dims maybe 1536 etc. using fft2 topk maybe okay.

Need produce script path. user asked new script; likely under /data/disk2/xby/Misc/HSVD maybe or TinySR/test. Considering project config says new commands and agents in .kilo only. For user code, place likely `TinySR/test/analyze_tinysr_layer_compression.py` maybe best near capture script. We'll create one script in TinySR/test.

Need inspect module forward signature of transformer and vae in capture script. we can reuse code from capture script. Since script in TinySR/test should use same functions for loading model and preprocessing. Could copy minimal.

Need gather layer selection from manifest. Candidate layers "suitable".
Could define suitability:
- module is nn.Linear.
- module is nn.Conv2d with kernel_size==(1,1) and groups==1.
- exclude tiny weights below min params threshold default 4096 maybe to avoid very small.
- optionally restrict names containing transformer and pointwise conv.
- Only layers that appear in manifest at least once.
- For each candidate we capture first input from replay to avoid repeated calls storing huge.
But some modules called multiple times maybe with different input distribution. Maybe we can gather from manifest call frequency and maybe capture up to k calls per layer and concat sample of each maybe heavy. We can capture first and maybe average over first few.
Could capture small sample per call:
- linear input maybe shape [1, seq, dim]. seq maybe up to 4096? manageable.
- conv activation huge [1,64,512,512] => flatten huge 16M rows, impossible. We'll sample random spatial tokens.
Could capture full input in pre-hook then sample quickly to smaller matrix and store CPU float32.
Implement function:
```
def sample_matrix_from_input(module, x, max_samples):
  if linear: x_flat = x.reshape(-1, in_features); sample rows if >max.
  if conv1x1: x_flat = x.permute(0,2,3,1).reshape(-1,in_channels); sample rows.
```
Need get input from pre-hook: inputs tuple first element x.
During forward, x on GPU maybe fp16. We can sample using torch.randperm maybe expensive on huge rows. Use stride sampling deterministic:
```
if rows > max_samples:
    step = rows / max_samples
    idx = (torch.arange(max_samples, device=x.device)*step).long()
    x_flat = x_flat[idx]
```
This avoids random overhead.
Need support capturing multiple calls per layer; maybe keep aggregator list and then concatenate up to cap_total_samples per layer across calls.
Pseudo:

```
layer_samples: dict[name, list[tensor_cpu]]
call_count: dict[name,int]
def pre_hook(module, inputs, name):
    x = inputs[0]
    x_mat = to_2d_samples(module, x, per_call_samples)
    if x_mat is None: return
    if name not in samples: samples[name] = []
    total = sum(s.shape[0] for s in samples[name])
    if total >= max_total_samples: return
    remain = max_total_samples - total
    if x_mat.size(0) > remain: x_mat = x_mat[:remain]
    samples[name].append(x_mat.cpu().float())
    call_count[name]+=1
```

Need layer output error compute with sampled X:
- For linear W [out,in], bias maybe [out].
`Y_orig = X @ W.t(); Y_comp = X @ W_comp.t()`
if bias included same cancels difference when relative error denominator includes Y_orig with bias maybe we should include to reflect actual layer output magnitude.
```
if bias is not None:
   Y_orig += bias
   Y_comp += bias
err = norm(Y_orig - Y_comp)/norm(Y_orig)
```
Need memory maybe output dims large; if X max 16384 and out maybe 1536 => ~25M ~100MB float32. Manage maybe maybe heavy multiple layers sequential. We'll compute one layer at time and free.

Compression methods:

- truncated_svd_torch(W, target_params):
Need derive rank from budget: r = target_params/(m+n). ensure >0.

- hsvd_torch + find_best? Could use functions from hsvd_validation maybe heavy due searching block sizes.
We can import from hsvd_validation? Since script located TinySR/test maybe import via path maybe easier to copy required functions maybe minimal adaptation.
Need maybe avoid dependency on hadamard etc. For layer analysis we can skip smooth/hadamard and just use raw W + X. We'll still run H-SVD by function from hsvd_validation maybe heavy and may fail for tiny dims.
Could implement simpler H-SVD search:
Given budget ratio p, target_params=int(p*orig_params)
We need compute best H-SVD config using output error with X samples for this layer only. Could run candidate r and block sizes.
Maybe simpler and faster:
- Choose r_svd = int(target/(m+n)); compute global Wg.
- For H-SVD we can call existing `find_best_hsvd_config_optimized` requiring max_r_global and maybe X.
It uses calculate_relative_error with X and tries candidate r and block sizes.
But this function expects W as tensor maybe on GPU and may perform expensive SVD for each layer.
Could import by loading file? not module package maybe path. easier copy relevant functions truncated_svd_torch, hsvd_torch, calculate_relative_error_torch, find_best... to script.
Need to ensure for conv1x1 where m,out and n,in maybe small.

Could for simpler use low-rank only? But user specifically "H-SVD". so implement full.

Need compute FFT-MixedQ4 adjustments.
Maybe for this script we can adjust to use `rfft2` to preserve realness and use budget cost based on original num params. Let's design a modified method pragmatic:

Given real W m x n, budget target_params (# float32 equivalent real values allowed, derived from ratio * m*n).
Method:
- Compute full frequency F = fft2(W) complex m,n.
- We'll maintain old cost model maybe comparing real-value count relative to original.
But for fair to target_params where total original params = m*n real values.
In old function, full precision frequency representation cost 2*m*n > original by 2x; then budgets 20-70% always below min_budget for q=4 maybe min_budget = m*n*0.25 -> okay. keep_fp solve maybe maybe.
But if target 20% maybe 0.2m n < min_budget(0.25mn) so fallback to top-k dropout occurs often. That's maybe not desirable. Could adjust q_cost to q/32 rather than 2*q/32 if budget measured in original real parameters (not complex real comps). Wait yes! In script old cost uses original parameters m*n as denominator. For complex coefficient two real numbers, they considered cost 2. But if baseline parameter is real weight (1 per element), mapping from frequency full complex to time-domain real maybe redundant due conjugate symmetry. So cost 2 overestimates by ~2x. Maybe better adjust for real matrices: treat full-precision complex coefficient effective cost 1 due symmetry; and q-bit cost q/32. This may allow budgets <25%. maybe needed.
Maybe this is "必要时调整" exactly.
Let's adjust method:

- Use `fft2` and keep symmetry maybe but if we keep all coefficients and transform back exactly.
- Define effective costs w.r.t original real parameter count:
  - fp_cost = 1.0
  - q_cost = high_freq_bits / 32.0
Because each real spatial parameter corresponds to one float32, and if we had perfect packed q-bit per spatial param cost q/32.
But in frequency domain this is conceptual budget not actual storage.
Then min_budget = total_coeffs * q_cost maybe 0.125mn for q4? wait q/32=0.125. maybe maybe.
This may be more realistic for comparing to original.
Need mention in output metadata.

Also to address conjugate symmetry, maybe implement `rfft2` variant where coeff count n_r and cost maybe still approx? But if cost model relative to original perhaps we can just avoid complexities by quantizing all coeff and ifft real part; won't break too much. But to ensure stable output maybe use `torch.fft.rfft2` + `irfft2` easier and always real. We'll do that maybe.

Design `fourier_spectral_decomposition_mixed_quant_real_budget(W, target_params, high_freq_bits=4)`:

1. `F = torch.fft.rfft2(W)` shape m,nr complex.
2. magnitude = abs(F)
3. N = m*nr
4. cost:
   - fp_cost = 1.0  # effective per rfft coeff ??? not exact.
   - q_cost = high_freq_bits/32.0
But N in rfft domain not equal original mn; with this cost full q cost = N*q/32 >? For n large, N~0.5mn. then min_budget around 0.0625mn for q4; too low maybe.
Could maybe convert using per-original coeff cost:
since rfft has half coeffs representing all info. each coeff corresponds to approx 2 spatial params? maybe not.
Simpler not to complicate; maybe keep full fft2 to avoid N mismatch.

Could keep fft2 but enforce Hermitian symmetry by averaging with conjugate counterpart after modification.
Need function to symmetrize easier using roll/flips:
For unshifted F shape m,n indices u,v. Pair index (-u mod m, -v mod n). We can create partner using index arrays:
```
u_idx = (-torch.arange(m, device=F.device)) % m
v_idx = (-torch.arange(n, device=F.device)) % n
F_conj_partner = torch.conj(F[u_idx][:, v_idx]) ??? 
```
Need correct broadcasting:
`F_partner = torch.conj(F.index_select(0, u_idx).index_select(1, v_idx))`
Then sym = 0.5*(F + F_partner)
```
def enforce_hermitian(F):
    u = (-torch.arange(F.shape[0], device=F.device)) % F.shape[0]
    v = (-torch.arange(F.shape[1], device=F.device)) % F.shape[1]
    partner = torch.conj(F.index_select(0,u).index_select(1,v))
    return 0.5*(F+partner)
```
This ensures hermitian. apply before ifft.
In shifted domain pair mapping changes due shift; easier operate unshifted without fftshift. We can skip shift and use top-k by magnitude flatten across all freq; no need shift.
Then we can enforce symmetry easily.
Algorithm:
- F = fft2(W) no shift.
- mag abs(F)
- total_coeffs = m*n
- cost model maybe old or adjusted.
Need avoid q cost min budget issue with 20% maybe fallback.
If keep old q_cost=2*(q/32)=0.25 for q4 and target >=0.2, first ratio 20% <0.25 so fallback top-k. maybe acceptable maybe but maybe to avoid constant fallback we adjust to real-budget costs.
Since user requested maybe adjust if needed; I'd adjust to real-budget cost:
`fp_cost=1.0`, `q_cost=high_freq_bits/32.0`.
This matches original real weight params.
Then keep_fp solving.
- threshold and mask in magnitude for keep_fp.
- high = F*(~mask) ; quant real/imag via symmetric quantize int and scale.
- F_mixed = F*mask + high_q
- enforce hermitian: F_mixed = enforce_hermitian(F_mixed)
- W_approx = ifft2(F_mixed).real
- actual_params_eff = keep_fp*fp_cost + (N-keep_fp)*q_cost
But if enforce hermitian changes mask maybe not exact budget but okay.

Potential issue mask may not symmetric leading hermitian averaging may alter fp/high quant assignments. Could sym mask too:
`mask_sym = mask | partner(mask)` by index_select.
Implement:
```
partner_mask = mask.index_select(0,u).index_select(1,v)
mask_fp = mask | partner_mask
```
then keep count maybe increases >keep_fp.
Let's implement simpler:
- select top-k unique by magnitude of pairs? complex.
Given analysis script approximate; can skip strict symmetric pair counts.
Maybe easier use rfft2 and avoid symmetry trouble; but cost mapping tricky.
Could still use rfft2 with cost relative to original `fp_cost = 2 * (m*n)/(m*nr) /?` hmm.

Let's maybe keep old method with shift simpler and adjust only cost model from 2 to1? But no guarantee realness due asymmetry; yet taking real part works; old method did that.
Given request "必要时调整" maybe we can adjust only budget model and support conv layer flatten; not heavy. mention in code comment.

Let's design new script focusing error analysis not method perfection. We'll include both default original and adjusted FFT cost mode; default adjusted maybe `--fft_real_budget` true.
Implementation steps:

- parse args: model paths maybe optional; capture_dir required.
- load run_meta args maybe use these defaults.
- budgets str.
- max layers, min params, max samples per layer.
- device.

Workflow:
1. Load manifest JSON.
2. Load run_meta args for model paths; allow overrides.
3. load models using same code as capture script.
4. Build candidate modules dict from model.named_modules with full names prefixed `transformer.` or `vae.`.
   Criteria:
   - isinstance(nn.Linear)
   - isinstance(nn.Conv2d) and kernel_size==(1,1) and groups==1
   - weight numel >= min_weight_params
   - present in manifest layers set
5. Register forward_pre_hook on selected layers to capture sampled inputs using one replay run replicating capture pipeline maybe from run_meta args input_image etc.
   Need run pipeline same as capture script (encode->transformer->decode).
   Because to capture transformer inputs maybe we need whole pipeline.
6. run one forward pass no grad.
7. remove hooks.
8. for each selected layer with samples:
   - X = concat sample tensors to device maybe float32.
   - get weight matrix:
     եթե Linear W=module.weight.detach().float(); bias same.
     եթե Conv1x1 W=module.weight[:, :,0,0].detach().float(); bias.
   - orig_params = W.numel()
   - for each budget ratio:
      target=int(ratio*orig_params)
      - SVD: rank = target/(m+n), W_svd from truncated_svd_torch; compute output error using layer type (linear-like with sampled X). store param_count
      - H-SVD: max_r=rank maybe >=1; call find_best_hsvd_config_optimized(W,target,max_r,min_block maybe 4?) ; if returns 0 fallback to svd maybe None.
        maybe min block size should adapt to dims at least 4.
      - FFT-MixedQ4: call adjusted function maybe default.
      compute output errors and weight errors.
9. aggregate result list and save json/csv maybe csv easiest.
- Save to out_dir maybe under capture_dir/analyze_compression.
Need generate summary ranking per method at each budget maybe average error weighted by layer params.
Compute weighted avg output error:
`sum(err*orig_params)/sum(orig_params)`.

10. print concise summary with top sensitive layers maybe by 50% budget.

Need create new script file likely `/data/disk2/xby/TinySR/test/analyze_tinysr_layer_compression.py`.

Need ensure we can use apply_patch add file. But instructions say prefer apply_patch single file yes.

Need maybe run syntax check. use bash `python -m py_compile ...` maybe.

Need read some model class to know conv module names etc maybe not needed.

Need check if `test/capture_all_activations.py` has helper functions we can import rather than copy maybe avoid duplication. We can import parse? parse_args not needed. Could import load_models and preprocess_image by from test.capture... maybe path issues. better copy minimal functions.

Need maintain script independent.

Need check `AutoencoderTiny` encode/decode on CPU may heavy but okay.

Need ensure hooking on modules with same names as manifest; full names same as capture script `vae.<module_name>` and `transformer.<module_name>`. We'll replicate.

Need parse manifest quickly huge 9336 lines; yes.

Need map candidate order maybe by weight numel desc and limit layers count maybe default 40.

Potential memory/time:
- candidate layers maybe many. capturing input sample in hooks for all candidate layers can still heavy because each hook sampling to CPU and storing maybe manageable with limits.
per layer max_total_samples default maybe 4096 and feature dims maybe 3072 => 50MB each; for many layers huge. Need limit dynamic.

Could store only one sample per layer maybe 1024 rows. for feature dim 3072 => 12MB each. 100 layers -> 1.2GB still high. We'll set default `max_samples_per_layer=1024`, `max_layers=24` selecting largest params. Good.
For conv1x1 with in maybe 256 => 1MB each.
Selecting top 24 by params manageable ~300MB CPU.

Maybe user may want all selected. We can include CLI to adjust.
Candidate selection:
- sorted by weight numel desc.
- default max_layers 24.
Given transformer many linear maybe about maybe 100. choose top 24.

Need output per-layer per-method/per-budget maybe JSON list huge but fine.
Let's craft data structure:

```
results = {
 "config": ...,
 "selected_layers":[...],
 "layers":[
   {
      "name":..., "type":"Linear","weight_shape":[...], "orig_params":...
      "sample_rows":int,
      "budgets":[
         {
           "ratio":0.2,
           "target_params":...
           "svd": {"param_count":..., "weight_rel_error":..., "output_rel_error":...},
           "hsvd": {... maybe null},
           "fft_mixedq4": {...}
         },...
      ]
   },...
 ],
 "summary": { "by_budget":[...]}
}
```

Need implement compute functions:

- `relative_error(a,b,eps=1e-12)`.

- `build_layer_matrix(module)` returns W2d,bias,type.
   Conv1x1: weight shape [out,in,1,1]
.

- `output_from_matrix(X,W,bias)` returns X@W.T + bias.

- `compute_errors(W_orig,W_approx,X,bias)` weight and output errors.

Need ensure tensors on same device. We'll run analysis on GPU maybe faster.
Could do calculations in float32 on device chosen. For CPU maybe slower.
Set compute device default from run_meta args maybe `cpu` from capture but user likely wants real maybe maybe. We'll default `cuda` if available else cpu and allow override.

load model weight dtype maybe from run_meta args maybe fp32; but for analysis maybe float32.

Selecting activation input image from run_meta input_image absolute path maybe okay.

Need reuse run_meta args for model paths. run_meta has args with keys names in capture script, plus maybe relative paths. Since we'll run with workdir TinySR root to resolve paths.
Script should set cwd to TinySR root.
Given script in TinySR/test, root is parents[1].

Could parse CLI overrides:
`--pretrained_model_name_or_path`, etc optional default None.

For simplicity if user passes capture_dir and not overrides, script uses run_meta args. If missing uses defaults from test.sh maybe.

Need include LoRA loading like capture script.

Need implement pipeline forward minimal from capture script:
```
model_input = vae.encode(pixel_values).latents*vae.config.scaling_factor
model_pred = transformer(hidden_states=model_input,timestep=timesteps,pooled_projections=pooled_prompt_embeds,return_dict=False)[0]
latent_stu = model_input - model_pred
_ = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0]
```
Need pooled embeds path from embedding_dir/pool_embeds.pt.

Need preprocess image. In capture script they resize to process_size/upscale etc. We'll copy.

Need sample conversion in pre-hook:

```
def extract_input_matrix(module, x, max_rows):
    if isinstance(module, nn.Linear):
      if x.shape[-1] != module.in_features: maybe transpose?
      x2 = x.reshape(-1, x.shape[-1])
    elif isinstance(module, nn.Conv2d) and module.kernel_size == (1,1) and module.groups ==1:
      if x.ndim !=4: return None
      x2 = x.permute(0,2,3,1).reshape(-1,x.shape[1])
    else: return None
    # sample row stride
```
Need handle dtype float16 etc -> float32.
`x = x.detach()`.

Potential for linear modules with 2D inputs maybe [N, d], works.

Should we capture activation from existing files directly? user requested use real activation data path. Our script replays model maybe still uses real data but not existing capture. Could we integrate existing capture by selecting layers from manifest path and ensure same call context. It uses run_meta and manifest yes.
Maybe we can also load manifest outputs to verify that layer was captured and call count. We can use that.
Add `manifest_counts` from file. Candidate only if count>0.

Could also optionally reuse manifest output for additional stats (activation shape). not needed.

Need adjust FFT mixed method:

I will implement `fourier_spectral_decomposition_mixed_quant` with option `real_budget=True`.
Algorithm similar to old but with effective cost.

```
def fourier_spectral_decomposition_mixed_quant(W,target_params,high_freq_bits=4,real_budget=True):
    m,n=W.shape
    total_coeffs=m*n
    if total_coeffs==0:return zeros,0.0
    F = torch.fft.fft2(W)
    mag=F.abs()
    if real_budget:
      fp_cost=1.0
      q_cost=float(high_freq_bits)/32.0
    else:
      fp_cost=2.0
      q_cost=2.0*(high_freq_bits/32.0)
    min_budget=total_coeffs*q_cost
    if target_params < min_budget:
      # top-k only
      num_coeffs = max(int(target_params / max(fp_cost,1e-8)),1)
      num_coeffs=min(num_coeffs,total_coeffs)
      th = torch.quantile(mag.flatten(), 1 - num_coeffs/total_coeffs)
      mask=mag>=th
      F_comp = F*mask
      W_approx=torch.fft.ifft2(F_comp).real
      return W_approx, float(mask.sum().item())*fp_cost
    denom = fp_cost-q_cost
    keep_fp = int((target_params - total_coeffs*q_cost)//denom) if denom>0 else 0
    keep_fp=max(0,min(keep_fp,total_coeffs))
    if keep_fp==0: th=inf ; mask_fp = zeros bool
...
```

Then quantize high as before.
Potential issue scale tensor maybe scalar maybe from entire tensor; yes.
Need quantization function returning int8+scale.

For top-k threshold with equal values may keep more than num_coeffs. acceptable.
We might include optional `enforce_hermitian` bool default True:

```
if enforce_hermitian:
   F_mixed = enforce_hermitian_symmetry(F_mixed)
```
Let's implement.

`enforce_hermitian_symmetry` with index_select partner.

```
def enforce_hermitian_symmetry(F):
   m,n=F.shape
   idx_m = (-torch.arange(m, device=F.device)) % m
   idx_n = (-torch.arange(n, device=F.device)) % n
   partner = torch.conj(F.index_select(0, idx_m).index_select(1, idx_n))
   return 0.5*(F+partner)
```
Need ensure special freq (self-conjugate) become real due averaging.

Good.

H-SVD function:
We can copy from hsvd_validation but maybe optimize. We'll include simplified search:

`find_best_hsvd_config(W,target_param_budget,max_r_global,min_block_size=8,X=None)` maybe.
Could use candidate r list unique [1,2,4,8,16,32,max_r//2,max_r]. block candidates powers of two from min to min(m,n)//2 maybe.
Since many layers, search heavy. We'll make quick:
- r_candidates = [max_r, max_r//2, max_r//4, ...]
- block candidates [8,16,24,32,48,64] <= min(m,n)
In each compute total params if <= target, evaluate output error with X.
Need compute W_approx maybe call hsvd_torch.

Given many layers, we need speed. We can maybe use only a few candidates.
Implementation:
```
def find_hsvd_config_fast(W,X,target,svd_rank):
   m,n = W.shape
   max_r = max(1,min(svd_rank,m,n))
   r_candidates = sorted(set([1,2,4,8,16,max_r//2,max_r]), reverse=True?) maybe.
   block_candidates = [8,16,24,32,48,64,96,128]; choose <=min(m,n)
   best=None
   for r in r_candidates:
      global_p = r*(m+n)
      rem=target-global_p
      if rem<=0: continue
      for s in block_candidates:
          if s>m or s>n: continue
          nbm=m//s; nbn=n//s
          if nbm==0 or nbn==0: continue
          local_p = nbm*nbn*(2*s)
          total = global_p+local_p
          if total>target: continue
          W_hat,_=hsvd_torch(W,r,(s,s))
          err=compute_output_error_matrix...
          update
```
Could be many computations but manageable for 24 layers*6 budgets* maybe 20 combos huge maybe impossible.
Need reduce complexity. Maybe for each layer-budget, computing hsvd multiple full SVD of matrix and block SVD heavy. Could slow extremely.
Could reuse global SVD per layer independent budget maybe helpful.
Alternative simpler: For H-SVD error analysis approximate with one config derived:
- set r_global = max(1, int(0.6 * target/(m+n)))?
- choose block size from budget formula maybe.
But user asked using H-SVD method maybe not exact best maybe okay.
Could use `find_best_hsvd_config_optimized` from existing script maybe optimized but still heavy per layer.
Maybe limit layers maybe top 12.

Better approach: For practical script, for each layer and budget we can compare:
- SVD baseline and FFT-MixedQ4 and H-SVD.
But H-SVD search heavy. Could reduce search to deterministic config:
1) Compute r_global = max(1, int(0.35*target/(m+n))) maybe allocate 35% budget to global branch?
2) remaining budget determines block size s from formula local_params=2mn/s => s≈2mn/remaining
round to multiple 8.
Then compute once.
This is not optimal but fast.
Let's derive:
local_params ≈ num_blocks_m*num_blocks_n*(2s) for square s dividing dims. if m,n large and divisible, approx (m/s)*(n/s)*(2s)=2mn/s.
So s ≈2mn/remaining.
Implement:
```
r_global = min(max(1, int(target/(3*(m+n)))?), min(m,n))
for maybe try few fractions [0.25,0.5,0.75] and pick best (3 tries). 
For each fraction:
 rg = min(max(1,int((target*frac)/(m+n))),min(m,n))
 rem=target-rg*(m+n)
 if rem<=0 continue
 s_est = int(max(4, round((2*m*n)/rem)))
 s = nearest_even_multiple(s_est,4) capped.
 compute hsvd if valid.
 pick best output error.
```
This avoids huge search.
Need still call hsvd_torch performing one full SVD of W and block batched SVD. For 24 layers*6*3=432 full SVD maybe heavy but maybe manageable with moderate dims maybe huge maybe 1536x1536 each heavy maybe too much. Could still be much.
Could cache W_global for each rg maybe within layer-budget combos. For each layer we have maybe rg values repeated across budgets? maybe not.
Given complexity maybe better to sample fewer layers. Could still run long but user asked script, not run now.
So fine.

Could keep function more robust but not insanely slow; user can set max layers.

Need output maybe JSON plus CSV.
Could write JSON only.

Need determine "suitable layers" extraction from existing activation manifest path. We'll use names present in manifest. include call counts maybe from manifest.

Potential mapping for module names:
`model.named_modules()` gives names without `vae.` prefix inside each model. We'll add prefix same as capture script.
Construct module map:
```
module_map[f"vae.{name}" if name else "vae"] = module
module_map[f"transformer.{name}" if name else "transformer"] = module
```
Candidate selection based on module type and manifest layer counts.

Need parse arguments of run_meta maybe includes relative paths; set cwd to TinySR root to resolve. script sets os.chdir(root).

Need implement LoRA loading requiring diffusers and peft packages.
Since existing capture script imports diffusers stable etc.

Need avoid using `find` etc.

Now create script with ~400 lines maybe.

Let's craft script carefully with functions:
- parse_args
- parse_budget_ratios
- load_capture_meta
- load_models
- preprocess_image
- gather_manifest_layer_counts
- collect_candidate_layers
- extract_input_samples
- run_replay_and_capture_inputs
- matrix compress methods functions

Implement compress methods from hsvd:
- truncated_svd_torch_matrix
- hsvd_torch_matrix
- fourier_mixedq4_adjusted
- maybe fallback to low-rank for small dims.

For hsvd_torch_matrix, ensure if block size <=0 etc.
Edge cases:
- if m<2 or n<2, skip hsvd.
- if num_blocks_m==0 or num_blocks_n==0 skip.
- if bh==1 and bw==1 local params huge? but can.
We'll enforce min block size 4.

`nearest_divisor_size(m,n,s_guess,min_block=4,max_block=128)`:
- candidate divisors maybe multiples of 4 from min..max and <=min(m,n).
- choose size that yields total_params <= budget and close to guess.
Simpler we can choose from candidate list [4,8,16,24,32,48,64,96,128].
and ensure >0 blocks.

`build_hsvd_configs`:
for each ratio allocate global fraction list [0.35,0.5,0.65].
for each compute rg and block candidate from remaining budget.
maybe one candidate.
Could also include few block sizes around guess [-8,0,+8] etc.
makes maybe <9 eval.

Let's implement find_best_hsvd_fast:
```
def find_best_hsvd_fast(W,X,bias,target_params,min_block=8):
 m,n=W.shape
 max_rank=min(m,n)
 best=None
 global_fracs=(0.35,0.5,0.65)
 block_candidates=[4,8,16,24,32,48,64,96,128]
 for gf in global_fracs:
   rg = int((target_params*gf)/(m+n))
   rg=max(1,min(rg,max_rank))
   gp=rg*(m+n)
   rem=target_params-gp
   if rem<=0: continue
   s_est=max(min_block, int((2*m*n)/max(rem,1)))
   s_near=sorted(block_candidates, key=lambda s: abs(s-s_est))
   for s in s_near[:4]:
      if s>m or s>n: continue
      nbm=m//s; nbn=n//s
      if nbm==0 or nbn==0: continue
      lp=nbm*nbn*(2*s)
      total=gp+lp
      if total>target_params: continue
      W_hat,_=hsvd_torch(W,rg,(s,s))
      err=...
      update
```
If no candidate, return None.

Need compute output error uses X matrix and bias.

Output error function:
```
def compute_layer_errors(W_orig,W_approx,X,bias):
   w_err = relerr(W_orig,W_approx)
   Y_orig = X @ W_orig.t()
   Y_approx = X @ W_approx.t()
   if bias is not None:
      Y_orig = Y_orig + bias.unsqueeze(0)
      Y_approx = Y_approx + bias.unsqueeze(0)
   y_err = relerr(Y_orig,Y_approx)
```
Could overflow memory if X rows huge. we cap rows.

Need sampling input matrix from hook:
For linear, if x shape [..., in]
`x2=x.reshape(-1, x.shape[-1])`.
For conv1x1 x shape [B,C,H,W].
`x2=x.permute(0,2,3,1).reshape(-1,C)`
Then sampling.
Use `max_samples_per_call` maybe 2048 and max_samples_per_layer maybe 8192.
To keep memory.

Will store CPU float32.

In hook, skip if no expected features maybe check x2.shape[1]==in_dim from module weight.
Need module type check.

Need object Candidate dataclass maybe simpler dict.

Let's craft candidate dict:
```
{
 'name':full_name,
 'module':module,
 'kind':'linear'/'conv1x1',
 'weight_shape':list(module.weight.shape),
 'orig_params': module.weight.numel(),
 'manifest_calls':count,
}
```
Select top max_layers by orig_params descending.

Need run capture pipeline with hooks only on selected modules.
pre-hook returns input.
Since transformer and vae same modules as loaded.

Implement:
```
def run_replay_capture(selected, transformer, vae, args, device, dtype, max_per_call,max_per_layer):
    samples = {name: []}
    rows_count = defaultdict(int)
    call_count = defaultdict(int)
    hooks=[]
    for item in selected:
      name=item["name"]; module=item["module"]
      def pre_hook(mod, inputs, layer_name=name, layer_kind=item["kind"]):
         if not inputs: return
         x=inputs[0]
         mat=extract_input_matrix(mod,x,layer_kind,max_per_call)
         if mat is None or mat.numel()==0: return
         remain=max_per_layer - rows_count[layer_name]
         if remain<=0: return
         if mat.shape[0]>remain: mat=mat[:remain]
         samples[layer_name].append(mat.cpu())
         rows_count[layer_name]+=mat.shape[0]
         call_count[layer_name]+=1
      hooks.append(module.register_forward_pre_hook(pre_hook))
...
```
Need closure capturing layer_kind etc.

Then run same forward as capture script requiring pooled_prompt_embeds and image preprocessing.

Need load settings from run_meta args:
`args_from_meta = run_meta['args']` keys include input_image etc.
But script args may override some.
Let's design parse args includes optional overrides; if not given use run_meta.
Could implement helper:
```
def pick_arg(cli_val, meta_args, key, default):
   return cli_val if cli_val is not None else meta_args.get(key, default)
```
Simplify include only essential.

script parse args:
- `--capture_dir` required default given path.
- `--out_json` optional
- `--device` optional
- `--mixed_precision` choices fp16/fp32 optional
- model path overrides etc.
- `--high_freq_bits`
- `--budget_ratios` default "0.2,0.3,0.4,0.5,0.6,0.7"
- `--max_layers` default 24
- `--min_weight_params` default 4096
- `--max_samples_per_call` 1024
- `--max_samples_per_layer` 4096
- `--hsvd_min_block` default 8
- `--fft_budget_mode` choices ["legacy_complex","real_param"] default real_param.
- `--fft_enforce_hermitian` bool flag default true maybe with --no
In argparse সহজ with action store_true etc.

Need implement model loader uses cache maybe not necessary.
From test script uses cache_dir optional default "checkpoint/huggingface". capture script not. We'll just copy from capture script no cache.
Need import diffusers stable pipeline etc.

Potential issue running on CPU and huge model heavy but possible.

Now do we need use Task tool? instructions say if Task available use proactively for focused subtasks. Could use explore agent to inspect repo quickly? we already did. maybe not mandatory but "proactively" maybe should use maybe but can ignore maybe okay though maybe should have used. But no requirement maybe.

Need add script file new. Use apply_patch Add File.

Let's craft script content carefully with ASCII. Chinese comments maybe okay but default ascii; we can do English comments to comply.

Let's draft script around 500 lines maybe.

Need include shebang maybe not necessary.

Let's craft script with dataclass Candidate maybe readability.

Pseudo structure:

```
import argparse, json, os, sys, time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusion3Pipeline

sys.path.append(...)

from models.tinysr.tinysd3 import TinySD3Transformer2DModel
from models.vae.autoencoder_tiny import AutoencoderTiny
from utils.util import load_lora_state_dict
```

Define dataclass candidate with module etc.

`def parse_ratios` etc.

Potential issue dataclass with module typed torch.nn.Module not serializable.

Let's implement simple dict to avoid dataclass.

Functions:

- `relative_error`
- `symmetric_uniform_quantize` returns int8 and scale
- `fourier_spectral_decomposition_topk`
- `enforce_hermitian_symmetry`
- `fourier_spectral_decomposition_mixed_quant` with budget_mode option.
Need topk fallback for low budget using same budget_mode maybe.

`fourier_spectral_decomposition_topk(W,target_params,fp_cost)`
`num_coeffs=max(1,min(int(target_params / max(fp_cost,1e-8)),m*n))` etc.

- `truncated_svd_torch`
- `hsvd_torch`
- `find_best_hsvd_config_fast` maybe takes X,bias,target,block sizes etc.

- `module_kind(module)` returns str or None.
- `build_weight_matrix(module, kind)` returns W2d,bias
- `extract_input_matrix(module, kind, x, max_rows)` returns sampled matrix

- `load_models(cfg,weight_dtype,device)` similar.

- preprocess_image(args)...

Need config object maybe dict.
During parse capture metadata:

```
with open(capture_dir/run_meta.json)...
meta_args = run_meta.get("args",{})
```
Then build runtime_cfg dict with keys from meta args or cli.

For input image path maybe from run_meta["input_image"] absolute. But if missing use meta_args input_image relative.
Need function to resolve path relative to TinySR root.

Set `repo_root = Path(__file__).resolve().parents[1]`
`os.chdir(repo_root)`.

`resolve_repo_path(p)`: if absolute return as is else repo_root/p.

Model paths in meta may be relative; from root.

Load manifest JSON:
count layers:
```
for item in manifest: layer_counts[item["layer"]] += 1
```

Collect candidate modules:
```
def collect_candidates(model, tag, layer_counts, min_params):
   for name,module in model.named_modules():
      if name=="" continue?
      full=f"{tag}.{name}" if name else tag
      if layer_counts.get(full,0)==0: continue
      kind=module_kind(module)
      if kind is None: continue
      params=module.weight.numel()
      if params<min_params: continue
      ...
```
Need include module pointer.
Then combine transformer and vae lists sort by params desc and maybe call count.
Take top max_layers.

Log selection print.

Capture inputs:
Need no grad and eval.
`pooled_prompt_embeds = torch.load(resolve_repo_path(cfg["embedding_dir"])/"pool_embeds.pt", map_location=device).to(dtype=weight_dtype)`
if path string maybe with trailing slash.

Preprocess with `cfg`.
then forward.
Need `torch.cuda.synchronize()` maybe if cuda for timing not necessary.

After capture evaluate results:
loop selected:
```
if not layer_samples[name]: continue
X = torch.cat(samples[name], dim=0).to(device=device,dtype=torch.float32)
W,bias=build_weight_matrix(...)
W=W.to(device,float32); bias...
orig_params = W.numel()
for ratio in ratios:
 target=max(1,int(ratio*orig_params))
 # SVD
 rank=max(1,min(int(target/(W.shape[0]+W.shape[1])), min(W.shape)))
 W_svd,svd_params = truncated_svd_torch(W, rank)
 werr,yerr=compute_errors
 # H-SVD
 hsvd_entry={"status":"skipped"...}
 cfg=find_best... maybe returns dict or None
 ...
 # FFT
 W_fft,fft_params = fourier...
```
Need compute for each method optionally catch exceptions to continue.
For h-svd function may fail on block reshape if dims not divisible; our hsvd handles by cropping; good.

`find_best_hsvd_config_fast` returns best dict with fields `r_global`,`block_size`,`param_count`,`W_approx` maybe huge; maybe just return cfg and W approx maybe heavy not required.
We'll compute inside and return tuple.

```
def find_best_hsvd_config_fast(W,X,bias,target_params,max_rank,min_block,block_candidates):
    ...
    best=None
    for ...:
        W_hat,param_count = hsvd_torch(W,r,(s,s))
        _, y_err = compute_errors(...)
        if better:
           best=(W_hat,param_count,r,s,y_err)
    return best
```
Need weight error compute afterwards from best W_hat.
Could compute once.

Potential performance: computing output error each candidate with X large; maybe heavy. we can maybe use smaller `hsvd_eval_samples` maybe from first 1024 rows of X for config search, then compute final on all rows for best.
Implement:
```
X_eval = X if X.shape[0] <= max_search_samples else X[:max_search_samples]
```
Pass both maybe.
Search errors on X_eval to speed.
Then final compute on X full.
Add arg `--hsvd_search_samples` default 1024.

For summaries:
After per-layer results, compute by budget:
for each ratio method list errors weighted by orig_params.
Need skip failed results.
```
for ratio_idx...
   weight_sum
   method sum = Σ(err*params)
```
Store.

Finally write output json to out path default capture_dir/compression_analysis.json.
Also maybe write CSV summary.

Let's craft generate `out_csv` summary lines with fields budget,method,weighted_output_rel_error,avg_weight_rel_error,total_layers_valid.
Could write using csv module.

Need create file with write tool? We can use apply_patch Add File. yes.

Need maybe run py_compile check using bash command. instructions say for file operations use read/write etc not bash. but can run python compile with bash yes.

Let's craft patch now.
ժ let's add script path. We'll use `/data/disk2/xby/TinySR/test/analyze_tinysr_layer_compression.py`.

Let's craft content carefully around 500 lines maybe.

Need ensure imports not unused maybe okay.

Let's craft script now.