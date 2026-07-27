Meta Review of Submission23990 by Area Chair mp1N
Meta Reviewby Area Chair mp1N21 Jul 2026, 15:52 (modified: 24 Jul 2026, 02:15)Senior Area Chairs, Area Chairs, Authors, Reviewers Submitted, Program Chairs, Area Chair mp1NRevisions
Metareview:
The paper does not have sufficient support from the reviewers to justify acceptance.

The main contribution of the work is algorithmic and theoretical. With my first skim of the paper, I think the theoretical results are of interest. However, when the main claim of the work is about more effective use test-time compute in diffusion models, it is completely legitmate that reviewers (and general ML readers) would be interested in wanting to see more conclusive experiments so they can put the paper in context of the related literature, or simply wanting to tell, if the proposed method actually "works" in the typical settings people use Best-of-N and other kinds test-time controlled generation methods.

Reviewers gave many useful pointers along the lines of how the experiments can be impiroved.

If the authors intend for the paper to be considered primarily on its theoretical merits, I could help steering the discussion in that direction. But it might be better for the authors to consider the option of revising the paper more carefully with stronger experiments and resubmit a much stronger version of the work to ICLR. Even if the paper gets in NeurIPS on the borderline, with weak experiments, it would be underselling imho.

Add:
Official Review of Submission23990 by Reviewer 8PHj
Official Reviewby Reviewer 8PHj03 Jul 2026, 15:05 (modified: 23 Jul 2026, 23:59)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer 8PHjRevisions
Summary:
This work addresses the challenge of how to allocate extra inference-time compute across the denoising trajectory of diffusion models. The paper proposes GAINS, a global scheduling method that decides how many noise candidates to evaluate at each timestep under a fixed NFE budget. The method uses offline profiling to estimate which timesteps are most sensitive to verifier-guided search, and online control to stop early on unproductive timesteps and reallocate saved compute later. The claims of the paper are supported by a clean two-level formulation, a water-filling-style theoretical analysis, and experiments on Stable Diffusion, EDM, and flow-based models showing improvements over uniform allocation on Brightness and Compressibility verifiers.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Quality
The paper is technically interesting and the problem formulation is sensible. I like the separation between local noise search and global scheduling; it makes the contribution easier to understand and makes the method potentially compatible with several existing local search operators. The offline-plus-online design is also well motivated: the offline stage captures average timestep sensitivity, while the online stage can recover some per-prompt variation by monitoring verifier-score gains and candidate variance.

The comparison set is also too narrow for the main claim. The main baseline is uniform allocation using the same ϵ-greedy noise search. This is a good ablation because it isolates the scheduling effect, but it does not establish competitiveness against the broader inference-time scaling literature. The paper itself lists many related methods but those are mostly classified rather than directly compared. For a paper whose novelty is global scheduling, the most important missing baselines are other adaptive or non-uniform schedulers, especially RBF [1.] and Verifier Threshold [2.].

The claims of the paper would be stronger if the authors included Offline Only results in the main scaling tables, not just the appendix ablation. Tables 1 and 2 report Uniform vs. full GAINS across NFE budgets for Stable Diffusion and EDM, but they do not show whether offline-only would already achieve most of the improvements. Without this, it is hard to tell whether the online controller is broadly important or whether a fixed offline profiled schedule is sufficient in most settings. A similar concern applies to Table 7. The operator-compatibility experiment shows that GAINS improves over uniform for zero-order and random search, but the gains are again modest. Since Table 7 does not include an offline-only variant, it is unclear whether these improvements come from the offline schedule alone or from the full offline-plus-online mechanism.

No qualitative examples to compare, only quantitive metrics are provided.

The theory is elegant but rests on assumptions that should be connected more carefully to practice. The location-scale result assumes smooth verifier-composed denoising, small stochastic perturbations, and a leading-order Taylor approximation. The offline optimality result further assumes timestep sensitivity is prompt-independent, which is exactly the condition that may fail for diverse prompt distributions.

Clarity
The paper is generally clear and well organized. The two-level framework is easy to follow, and Figure 1 helps communicate the main idea: a global scheduler allocates per-step budgets, and local operators consume those budgets by evaluating multiple noise candidates. Algorithm 1 is also useful because it makes the online control rule concrete.

Significance
The paper’s significance is strongest as a resource-allocation framework, but the comparison with existing approaches is missing, so its not clear how good as compared to existing works.
2, The empirical evaluation mainly relies on Brightness and Compressibility as verifiers. These are reasonable controlled objectives for testing whether the scheduler can improve a target score, but they are not strong proxies for broader notions of image quality, prompt alignment, realism, or human preference. As a result, the current experiments show that GAINS can better optimize the chosen verifier, but they do not rule out the possibility that the method is simply learning where those particular metrics are easiest to exploit. I understand that evaluating human preference, compositional alignment, counting, OCR/text rendering, or other semantic benchmarks may be outside the main scope of this work. Still, this narrows how broadly the results should be interpreted and how directly future work can build on the method for general-purpose image generation. The claims of the paper would be stronger if at least some independent alignment or preference metrics were included alongside the optimized verifier scores.

Originality
The paper is original in its framing of inference-time diffusion search as a global per-timestep budget-allocation problem. Prior work has studied local noise search, Best-of-N sampling, zero-order optimization, particle methods, and adaptive search, but this paper’s contribution is to make the global schedule itself the object of study. The MDP/table in the appendix is a useful way to position GAINS among related methods.

That said, the paper should do more to distinguish itself empirically from the closest adaptive-scheduling methods. The appendix lists RBF as “online rollover” and VT as “online threshold,” both of which seem very close in spirit to GAINS. Without direct comparisons to these methods, the originality is clearer at the formulation/theory level than at the empirical performance level.

===References=== [1.] VERIFIER THRESHOLD: AN EFFICIENT TEST-TIME SCALING APPROACH FOR IMAGE GENERATION

[2.] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing

Quality: 2: not good
Clarity: 3: good
Significance: 3: good
Originality: 3: good
Questions:
A useful addition would be to report Offline Only / Full GAINS across all main tables, together with a discussion of when the online controller gives meaningful gains and when offline profiling is already sufficient.

Compare with competitive methods such as RBF [1.] and Verifier Threshold [2.].

Provide some qualitative examples (probably not feasible in the rebuttal, given the space is limited).

Suggestion
Minor comment: Please consider adding clickable hyperlinks for appendix references. For example, in Line 324, “Four stress tests (full tables in Appendix D)” should link directly to Appendix D, so that clicking “Appendix D” takes the reader to the corresponding section. This would improve navigation.

===References=== [1.] VERIFIER THRESHOLD: AN EFFICIENT TEST-TIME SCALING APPROACH FOR IMAGE GENERATION

[2.] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing

Limitations:
yes

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
No

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Official Review of Submission23990 by Reviewer GLgQ
Official Reviewby Reviewer GLgQ28 Jun 2026, 12:55 (modified: 23 Jul 2026, 23:59)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer GLgQRevisions
Summary:
The paper studies inference-time scaling for diffusion models in the regime where additional compute is spent on within-step noise search. The within-step noise search evaluates multiple noise candidates at each denoising timestep and selects the candidate with the highest verifier score. The different timesteps of denoising have different sensitivities to noise refinement. Therefore, the paper formulates global noise trajectory search as a budget allocation problem: given a denoising trajectory of length 
 and a total budget 
, determine the number of local-search iterations 
 at each timestep so that 
 while maximizing the expected verifier score of the final sample. To solve this problem, the authors propose GAINS, which combines offline sensitivity profiling with online feedback control. The offline stage is based on the average verifier-score improvement obtained from additional local-search iterations on a calibration set of prompts. The online stage adapts the offline allocation based on observed verifier-score improvements and score variance (higher variance indicates higher potential of finding the improvement with a new noise candidate). The theoretical analysis of the optimal offline allocation shows that a timestep receives more budget when it is more sensitive to noise refinement. The paper also establishes a worst-case regret lower bound for purely online allocation policies, providing theoretical justification for the proposed hybrid offline-online design. Experiments on Stable Diffusion, EDM, and flow-based models with Brightness and Compressibility verifiers show that GAINS consistently improves over uniform allocation.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths
The theoretical results provide strong motivation for the proposed two-level approach. In particular, the regret lower bound justifies the need for offline allocation in addition to online adaptation. Theorems 5 and 6 provide a useful characterization of the optimal offline allocation, showing that timesteps with higher sensitivity to noise refinement should receive a larger share of the computation budget.
The modular separation between local search operators and global scheduling is also interesting. This decomposition is practically appealing, as advances in local search methods can be readily combined with improved global scheduling strategies.
Weaknesses
The experimental analysis of the paper is relatively weak. Even though the paper has interesting theoretical results, these results are primarily motivational. Moreover, the empirical evidence is not sufficiently strong to fully support the paper's claims.
The primary baseline considered in the paper is uniform allocation across denoising timesteps. It would be more convincing to compare against global scheduling strategies used in prior verifier-based inference-time methods and in general other inference-time scaling approaches (Ramesh and Mamdani 2025, Kim et al. 25), such as particle- or tree-based trajectory search approaches.
The paper only considers verifier scores as an evaluation metric. The paper does not evaluate perceptual quality or text-image alignment metrics. Without improvements on these complementary metrics, it is difficult to conclude that the proposed method improves overall image quality rather than simply optimizing the verifier.
The empirical gains are also fairly modest in several settings, particularly when combined with the zero-order and random local search operators, as well as on flow-based models. The proposed method further introduces several additional hyperparameters for online control (e.g., 
), as well as an offline profiling stage on a calibration set. Given this additional complexity, the empirical improvements appear somewhat limited.
The evaluation is conducted on a relatively small benchmark consisting of only 20 prompts with 10 repetitions. This raises concerns about the generality of the reported improvements.
It would be good to include an ablation study analyzing the contribution of the two signals used by the online controller: verifier-score improvement and candidate-score variance. In particular, it would be useful to evaluate variants that use only verifier-score improvements, only score variance, and both signals together
The writing is difficult to follow in some parts of the paper. For example, Section 4.3 presents the theoretical results in considerable technical detail, but provides limited discussion of their broader implications and intuitions. A discussion on broader implications and intuitions would improve readability.
Quality: 2: not good
Clarity: 2: not good
Significance: 2: not good
Originality: 2: not good
Questions:
Please see above.

Limitations:
The paper very briefly discusses the limitations.

Rating: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
No

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Official Review of Submission23990 by Reviewer x7rM
Official Reviewby Reviewer x7rM24 Jun 2026, 20:54 (modified: 23 Jul 2026, 23:59)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer x7rMRevisions
Summary:
The paper studies inference-time compute allocation for diffusion models. The main idea is to spend extra denoising evaluations non-uniformly across timesteps, instead of using the same search budget at every step. The authors propose GAINS, which combines offline timestep sensitivity profiling with an online controller that can stop early at less useful timesteps and reallocate budget.

The paper also gives a theoretical model suggesting that the benefit of extra candidates at a timestep depends on a timestep sensitivity factor and a sample-size factor. This leads to a water-filling-style offline allocation rule and a worst-case regret lower bound for purely online allocation.

Empirically, the paper compares GAINS against uniform allocation on Stable Diffusion, EDM, and a flow-based model. The reported gains are mainly on two simple verifier scores, Brightness and Compressibility.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths:

Interesting high-level idea. I like the question of where to spend inference-time compute along a diffusion trajectory. It is a natural complement to work that focuses on how to search within a single denoising step.

Clean two-level decomposition. The separation between a local search operator and a global scheduler is useful. It makes the method modular and helps position GAINS relative to uniform search, best-of-N, and other inference-time noise search methods.

The theoretical model gives useful intuition. The water-filling interpretation is a nice way to explain why non-uniform allocation can help when timestep sensitivities vary. The regret lower bound also gives a reasonable motivation for combining offline profiling with online control.

The paper includes several experimental settings. The authors test on Stable Diffusion, EDM, and a flow-based model, and they include ablations for online vs. offline-only scheduling and operator compatibility.

Weaknesses:

The empirical validation is too narrow for the claims. My main concern is that the experiments mostly optimize and evaluate Brightness and Compressibility. These are simple deterministic verifiers, but they are not the same as image quality, semantic alignment, aesthetics, diversity, or human preference. As a result, the experiments show that GAINS can improve these specific verifier scores, but they do not convincingly show that the method improves diffusion sampling quality in the broader sense.

There is not enough qualitative or visual evidence. For a diffusion-model paper, I would expect generated image grids, qualitative comparisons, and ideally examples showing when GAINS changes global structure or improves the sample in a meaningful way. The main paper mainly reports tables of verifier scores. This makes it hard to judge whether the gains correspond to visually or semantically better samples, or mainly to better exploitation of the chosen verifier.

The comparison baselines are limited. The main comparison is against uniform allocation with the same local search operator. This is a useful baseline, but not enough to establish that GAINS is a strong global scheduler. I would like to see comparisons against simple tuned non-uniform schedules, such as early-window, middle-window, late-window, oracle-profiled schedules on a held-out set, or learned/amortized per-step compute policies.

The theory is idealized relative to the actual algorithm. The theoretical model relies on a leading-order sensitivity approximation and a simplified random-search setting. The implemented GAINS controller uses offline profiling, empirical variance, recent gain thresholds, and early stopping. These are reasonable design choices, but the connection between the water-filling theorem and the actual controller is not tight enough. The paper should be clearer about which parts are theoretically justified and which parts are heuristic.

The offline profiling step needs more stress testing. GAINS depends on a sensitivity profile estimated from a calibration set. I would like to know how stable this profile is across prompt distributions, verifiers, and model families. If the profile is highly verifier- or dataset-specific, the practical usefulness of the method is more limited.

The claims about compute savings are verifier-dependent. The paper reports 20-50% fewer NFE for matching verifier scores, but since the verifiers are simple, it is unclear whether the same compute savings would hold for more realistic quality objectives such as CLIP alignment, aesthetic scores, reward models, or human preferences.

Quality: 2: not good
Clarity: 3: good
Significance: 2: not good
Originality: 3: good
Questions:
Can you evaluate GAINS on more realistic image-quality or alignment metrics, such as CLIP score, aesthetic score, ImageReward/HPS-style preference models, FID-like distributional metrics, or human preference?

Can you include qualitative image grids comparing uniform allocation and GAINS at the same NFE budget?

How sensitive is the offline sensitivity profile to the prompt distribution? Does a profile learned on one prompt set transfer to another?

Can you compare against simple tuned non-uniform schedules, such as early-only, middle-only, or held-out oracle-profiled schedules?

Which parts of the online controller are directly supported by the theory, and which are heuristic choices?

Does GAINS preserve sample diversity, or does optimizing a fixed verifier reduce diversity or overfit to the verifier?

Limitations:
The main limitation is that the empirical evidence is currently tied to simple verifier scores. This is a useful controlled setting, but it does not establish that GAINS improves general sample quality, semantic alignment, or human preference.

A second limitation is that the paper has little qualitative evidence. For a diffusion paper, visual comparisons are important, especially when the claimed improvement is about generation quality.

Finally, the theoretical analysis gives useful intuition, but it is more idealized than the implemented GAINS controller. I think the paper should more clearly separate the theoretically justified allocation model from the practical heuristic controller.

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
no formatting concerns

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
