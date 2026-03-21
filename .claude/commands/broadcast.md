# Broadcast - Intelligent Merge-Based Deployment to Target Projects

Deploy updated `.claude/` files to target projects with **diff-aware merge** instead of blind overwrite. Uses LLM subagents for files that need intelligent merging.

Supports **bidirectional sync**: discovers improvements in target projects and merges them back into the source before broadcasting.

## Arguments

`$ARGUMENTS` — Mode of operation. Examples:

| Argument | Action |
|----------|--------|
| (empty) or `all` | Deploy all categories to all targets (push-only) |
| `sync` | **Bidirectional**: pull improvements from targets first, then broadcast |
| `pull` | **Reverse-only**: scan targets for improvements, merge back to source, no broadcast |
| `commands` | Only `.claude/commands/*.md` |
| `shared` | Only `.claude/commands/_shared/*.md` |
| `standards` | Only `.claude/standards/*.md` |
| `rag` | Only `.claude/writing_references/**/*.md` |
| `templates` | Only `.claude/commands/templates/**/*.md` |
| `agents` | Only `.claude/agents/*.md` |
| `status` | Show target reachability and diff summary |
| `--dry-run` (suffix) | Preview changes without writing (e.g., `sync --dry-run`) |
| `add [path] [name]` | Add a new target to broadcast_targets.json |
| `remove [name]` | Remove a target from broadcast_targets.json |

**Category modifiers work with sync/pull**: `sync commands`, `pull shared`, etc.

## Protocol

### Step 0: Load Configuration

Read `.claude/broadcast_targets.json` from the source project root.

Parse `$ARGUMENTS` to extract:
- **mode**: one of `push` (default), `sync`, `pull`, `status`, `add`, `remove`
- **category**: one of `all`, `commands`, `shared`, `standards`, `rag`, `templates`, `agents`
- **dry_run**: whether `--dry-run` flag is present

If the config file is missing, report an error and stop.

### Step 1: Handle Non-Deploy Modes

**If `status`**: For each target in the config:
1. Check if the directory exists (use Bash `test -d`)
2. Report reachable vs. missing targets
3. For reachable targets, count files that differ from source
4. Display summary table

**If `add [path] [name]`**:
1. Read the JSON config
2. Add a new target entry with the given path and name
3. Default categories: `["commands", "shared", "standards"]`
4. Default exclude: `["theory-refs-*.md", "theory.md"]`
5. Write updated JSON
6. Report success

**If `remove [name]`**:
1. Read the JSON config
2. Find and remove the target matching the name
3. Write updated JSON
4. Report success

### Step 2: Resolve File List

For each selected category (or all categories if `all`):
1. If the category has a `glob` field → use Glob tool to find matching source files
2. If the category has a `files` field → use the explicit file list directly
3. Collect all matching source files

### Step 3: For Each Target Project

Check which categories this target subscribes to (from `categories` array).
Check which files are excluded (from `exclude` array — supports glob patterns).

For each file in the resolved file list:
1. **Skip** if the file matches any exclude pattern for this target
2. **Skip** if the file's category isn't in this target's `categories`
3. Compute the target file path: `{target.path}/{relative_path}`

### Step 4: Classify Each File

For each source->target file pair:

1. **If target file doesn't exist** -> action = `copy`
2. **If target file exists**, read both files:
   - If **identical** -> action = `skip` (already up to date)
   - If **different** -> determine merge policy:
     a. Check `policy_overrides` for an exact or glob match on the relative path
     b. Fall back to the category's `default_policy`
     c. Apply policy:
        - `source-wins` -> action = `overwrite`
        - `target-wins` -> action = `skip` (preserve target)
        - `merge` -> action = `merge` (needs LLM subagent)
        - `merge-additive` -> action = `merge-additive` (needs LLM subagent)

---

## Phase A: Reverse Discovery (sync/pull modes only)

**Skip this entire phase if mode is `push` (default).**

Before broadcasting, scan targets for improvements that should be pulled back into the source.

### Step A.1: Identify Candidates

From Step 4, collect all file pairs where:
- Target file **exists** AND **differs** from source
- Policy is NOT `target-wins` (those are intentionally project-specific)

Group candidates by source file path (one source file may differ across multiple targets).

### Step A.2: Evaluate Improvements

For each source file with 1+ differing targets, spawn a **haiku** Task subagent:

```
You are comparing versions of a file to identify improvements in the TARGET version(s) that are missing from SOURCE.

SOURCE (canonical version from the main project):
---
{source_content}
---

TARGET ({target.name} — project-specific version):
---
{target_content}
---

Analyze the TARGET for content that is MORE ADVANCED than SOURCE. Look for:
1. New features, sections, or capabilities not in SOURCE
2. Bug fixes or corrections to errors in SOURCE
3. Better examples, clearer explanations, or refined logic
4. Project-specific patterns that generalize well to other projects

For each improvement found, output a JSON object:
{
  "has_improvements": true/false,
  "improvements": [
    {
      "type": "new_feature" | "bug_fix" | "refinement" | "generalizable_pattern",
      "description": "Brief description of the improvement",
      "severity": "high" | "medium" | "low",
      "lines": "approximate location in target"
    }
  ],
  "recommendation": "pull" | "skip",
  "reason": "Why this should/shouldn't be merged back"
}

Rules:
- Project-specific paths, names, or configuration are NOT improvements (skip those)
- Content that references project-specific concepts is NOT generalizable
- New reusable features, protocols, or patterns ARE improvements
- Bug fixes and corrections are ALWAYS improvements
```

If multiple targets have the same source file with improvements, evaluate each target separately.

### Step A.3: Back-Merge

For files where at least one target has `recommendation: "pull"`:

1. **Report discoveries** to the user:
   ```
   REVERSE DISCOVERY: Found improvements in targets

   .claude/commands/polish-paper.md:
     [LLM_serving] HIGH: New L1 "Advisor Rules" priority level
     [LLM_serving] MEDIUM: Expanded modification categories (17 vs 7)

   .claude/commands/_shared/regression_guard.md:
     [LLM-for-BAI] LOW: Additional constraint type for symbol renames
   ```

2. **If `--dry-run`**: Stop here (just report, don't merge).

3. **If live run**: For each file with improvements, spawn a **sonnet** Task subagent to merge improvements back into the source:

```
You are merging improvements FROM a target project BACK into the canonical source.

SOURCE (canonical — this is the file you're improving):
---
{source_content}
---

TARGET ({target.name} — has improvements to pull back):
---
{target_content}
---

IMPROVEMENTS TO PULL:
{list of improvements from Step A.2}

MERGE RULES:
- Add the identified improvements to SOURCE
- Keep SOURCE's structure and organization as the base
- Do NOT add project-specific content (paths, names, local config)
- Generalize project-specific patterns where possible
- If improvement conflicts with SOURCE content, prefer the improvement (it's newer)

Output ONLY the merged source file content, no commentary.
```

4. **Write the improved source file** using the Write tool.
5. **Re-read the updated source** so subsequent broadcast uses the improved version.

### Step A.4: Pull Summary

```
=========================================
PULL COMPLETE  [category: {category}]
=========================================
Files scanned:     {total_differing}
Improvements found: {files_with_improvements} files
Back-merged:       {merged_back} files
Skipped:           {skipped} (project-specific only)
Mode:              {live | dry-run}
=========================================
```

**If mode is `pull`**: Stop here. Do not proceed to broadcast.
**If mode is `sync`**: Continue to Phase B (broadcast with updated source).

---

## Phase B: Broadcast (push and sync modes)

### Step 5: Execute Actions

**If `--dry-run`**: Do NOT write any files. Instead, collect all actions and display them in the report (Step 7).

**If live run**:

For `copy` and `overwrite` actions:
- Ensure parent directory exists (use Bash `mkdir -p`)
- Use the Write tool to write the source content to the target path

For `merge` actions:
- Spawn a **haiku** Task subagent with this prompt:

```
You are merging two versions of a configuration/protocol file.

SOURCE (canonical, from the main project):
---
{source_content}
---

TARGET (project-specific version, may have local customizations):
---
{target_content}
---

MERGE POLICY: merge
- Apply all structural updates, new sections, and corrections from SOURCE
- Preserve any project-specific customizations in TARGET (local hooks, project-specific paths, custom rules)
- If both versions modified the same section, prefer SOURCE structure but keep TARGET's local additions
- Do NOT remove content from TARGET that doesn't exist in SOURCE — it may be a local customization

Output ONLY the merged file content, no commentary.
```

For `merge-additive` actions:
- Spawn a **haiku** Task subagent with this prompt:

```
You are merging two versions of a writing reference file (sentence patterns, paragraph templates, or phrase collections).

SOURCE (canonical, may have new patterns added):
---
{source_content}
---

TARGET (project-specific version, may have its own patterns):
---
{target_content}
---

MERGE POLICY: merge-additive
- Add any patterns from SOURCE that don't exist in TARGET
- NEVER remove patterns from TARGET — they may be project-specific additions
- Preserve TARGET's ordering and organization
- If SOURCE has new sections or categories, append them
- Deduplicate exact matches only

Output ONLY the merged file content, no commentary.
```

After receiving the merged content from the subagent, write it to the target path using the Write tool.

### Step 6: Track Results

Maintain counters per target:
- `copied`: files that didn't exist at target and were written
- `overwritten`: files where source-wins replaced a different version
- `skipped_identical`: files that were already up to date
- `skipped_policy`: files preserved due to target-wins policy
- `skipped_exclude`: files excluded by project config
- `merged`: files that went through LLM merge
- `errors`: any failures

### Step 7: Report

Display a summary for each target, then a grand total.

**Per-target format** (only show targets with at least one action):

```
[target.name]
  copied: N  overwritten: M  merged: K  skipped: S  errors: E
```

**Grand total format**:

```
=========================================
BROADCAST COMPLETE  [category: {category}]
=========================================
Targets:    {reachable} reachable / {total} configured ({missing} missing)
Copied:     {total_copied}
Overwritten: {total_overwritten}
Merged:     {total_merged}  (LLM subagent)
Skipped:    {total_skipped}  ({identical} identical, {policy} target-wins, {exclude} excluded)
Errors:     {total_errors}
Mode:       {live | dry-run}
{If sync mode:}
Pulled back: {total_pulled_back} improvements from {targets_with_improvements} targets
=========================================
```

If `--dry-run`, prepend `[DRY RUN]` to the header and list each planned action:

```
[DRY RUN] Would copy:      .claude/commands/do.md -> LLM-for-BAI
[DRY RUN] Would overwrite:  .claude/standards/coding_standards.md -> LLM-for-BAI
[DRY RUN] Would merge:      .claude/commands/_shared/unified_protocol.md -> LLM-for-BAI
[DRY RUN] Would skip:       .claude/writing_references/_rag_log.md -> LLM-for-BAI (target-wins)
```

## Important Notes

- **Never overwrite `_rag_log.md`** — it's always target-wins (project-specific miss log)
- **Never overwrite `paper_state/*.md`** template instances — they contain project metadata
- **Parallel subagents**: When multiple files need merge for the same target, batch them into a single subagent call if possible (list all file pairs in one prompt)
- **Error recovery**: If a merge subagent fails, log the error and skip that file (don't fall back to overwrite)
- **Large targets**: Process targets sequentially to avoid overwhelming the filesystem, but merge subagents for different targets can run in parallel
- **Pull safety**: Back-merged improvements are written to the source only AFTER subagent approval. If the subagent produces garbled output, skip that file and report an error.
- **sync = pull + push**: `sync` is exactly `pull` followed by `push`, using the updated source files. No extra arguments needed.

## Begin

Parse `$ARGUMENTS` and execute the workflow above. Start by reading `.claude/broadcast_targets.json`.

- Default (no mode keyword): **push** — one-directional broadcast
- `sync`: **pull** improvements from targets first, then **push** updated source to all targets
- `pull`: **reverse-only** — discover and back-merge improvements, no broadcasting
