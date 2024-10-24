# Code Tracing of the GPU schedule rule

In TVM, there are quite a few scheduler techniques, for example auto-scheduler, autotvm, Relax's metascheduler and dlight scheduler. 

the diference between those are in the table below:
<table>
  <tr>
    <th style="background-color: #ADD8E6;">Feature</th>
    <th style="background-color: #B0E0E6;">AutoScheduler(Ansor)</th>
    <th style="background-color: #ADD8E6;">AutoTVM</th>
    <th style="background-color: #B0E0E6;">MetaSchedule</th>
    <th style="background-color: #ADD8E6;">DLight Scheduler</th>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">Template requirement</td>
    <td style="background-color: #F0FFFF; text-align: center;">No (fully automated)</td>
    <td style="background-color: #E0FFFF; text-align: center;">Yes</td>
    <td style="background-color: #F0FFFF; text-align: center;">No (high-level modular)</td>
    <td style="background-color: #E0FFFF; text-align: center;">No (pattern-driven)</td>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">Optimization type</td>
    <td style="background-color: #F0FFFF; text-align: center;">Search-based</td>
    <td style="background-color: #E0FFFF; text-align: center;">Parameter tuning</td>
    <td style="background-color: #F0FFFF; text-align: center;">Unified search space</td>
    <td style="background-color: #E0FFFF; text-align: center;">Lightweight scheduling</td>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">Dynamic workload support</td>
    <td style="background-color: #F0FFFF; text-align: center;">Excellent</td>
    <td style="background-color: #E0FFFF; text-align: center;">Limited</td>
    <td style="background-color: #F0FFFF; text-align: center;">Excellent</td>
    <td style="background-color: #E0FFFF; text-align: center;">Good</td>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">Target hardware</td>
    <td style="background-color: #F0FFFF; text-align: center;">General-purpose</td>
    <td style="background-color: #E0FFFF; text-align: center;">Target-specific</td>
    <td style="background-color: #F0FFFF; text-align: center;">General-purpose</td>
    <td style="background-color: #E0FFFF; text-align: center;">Accelerators (GPUs, TPUs)</td>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">Ease of use</td>
    <td style="background-color: #F0FFFF; text-align: center;">Medium</td>
    <td style="background-color: #E0FFFF; text-align: center;">Complex</td>
    <td style="background-color: #F0FFFF; text-align: center;">Medium</td>
    <td style="background-color: #E0FFFF; text-align: center;">Easy</td>
  </tr>
  <tr>
    <td style="background-color: #E0FFFF;">Best for</td>
    <td style="background-color: #F0FFFF; text-align: center;">Irregular workloads and new operators</td>
    <td style="background-color: #E0FFFF; text-align: center;">Known operators and static models</td>
    <td style="background-color: #F0FFFF; text-align: center;">Various models (static & dynamic)</td>
    <td style="background-color: #E0FFFF; text-align: center;">Large models, GPU workloads</td>
  </tr>
</table>

**MLC-LLM uses the DLight scheduler to schedule its LLM models.** The DLight scheduler applies different rules depending on the operator being optimized, such as matrix-vector multiplication, matrix-matrix multiplication, reduction, fallback rules, RMSNorm, transpose, etc.

**_Focus on Matrix Multiplication Scheduling Rules_**
For matrix multiplication, specific scheduling rules are applied based on the target GPU architecture and data type. The key rules include:

**Target-specific rules:**

* **Metal GPU:** Optimized for Apple’s Metal API.
* **CUDA GPU:** Specialized rules for NVIDIA GPUs using CUDA.
    * Within CUDA scheduling, there are two specific rules based on the data type:
        * **INT8:** Optimized for quantized models.
        * **Float16/Float32:** Optimized for half-precision and single-precision operations.
  
* **General scheduling rule:**
    * If the target device is neither Metal nor CUDA, a fallback scheduling rule is applied. This general rule ensures compatibility across other hardware platforms, though it may not be as optimized as the target-specific ones.

