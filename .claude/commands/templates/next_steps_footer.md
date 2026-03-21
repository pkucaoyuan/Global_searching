# Next Steps Footer Template

This template is appended to ALL check command outputs.

---

## Standard Footer Format

```
═══════════════════════════════════════════════════════════════════
                         NEXT STEPS
═══════════════════════════════════════════════════════════════════

Based on {N} issues found:

🔴 IMMEDIATE ACTIONS:
   {context-specific actions}

🛠️ RECOMMENDED COMMANDS:
   /command-1                  → Description
   /command-2                  → Description
   /paper-pipeline status      → See overall progress

📋 FULL WORKFLOW REMINDER:
   L0 Content    → /check-content-redundancy, /check-content-placement
   L1 Structure  → /check-paper-flow
   L2 Consistency→ /check-paper-consistency, /check-term-consistency, /check-cross-references
   L3 Style      → /check-ms-style, /check-or-style, /check-ml-style
   L4 Language   → /polish-paper

   Current level: L{X} | Next level: L{X+1}
```

---

## Context-Specific Next Steps

### After /check-paper-consistency (L2)

If issues found:
```
🔴 IMMEDIATE ACTIONS:
   1. Fix symbol conflicts in: [locations]
   2. Standardize notation for: [symbols]

🛠️ RECOMMENDED COMMANDS:
   [Fix issues manually first, then:]
   /check-paper-consistency    → Verify fixes
   /check-term-consistency     → Check terminology (same level)
   /check-cross-references     → Check references (same level)

   [When L2 passes:]
   /check-ms-style             → Move to L3 (venue style)
```

If no issues:
```
✅ L2 CONSISTENCY PASSED

🛠️ RECOMMENDED COMMANDS:
   /check-term-consistency     → Complete L2 checks
   /check-cross-references     → Complete L2 checks

   [If all L2 checks pass:]
   /check-ms-style             → Move to L3 (venue style)
```

---

### After /check-term-consistency (L2)

If issues found:
```
🔴 IMMEDIATE ACTIONS:
   1. Choose canonical term for each concept
   2. Search & replace in all files

🛠️ RECOMMENDED COMMANDS:
   [After fixing:]
   /check-term-consistency     → Verify fixes
   /update-paper-state [name]  → Sync terminology to docs

   [When L2 passes:]
   /check-ms-style             → Move to L3
```

---

### After /check-cross-references (L2)

If issues found:
```
🔴 IMMEDIATE ACTIONS:
   1. Fix broken refs: [list]
   2. Update numerical values: [list]

🛠️ RECOMMENDED COMMANDS:
   [After fixing:]
   /check-cross-references     → Verify all refs valid
   /update-paper-state [name]  → Sync cross_references.md
```

---

### After /check-content-redundancy (L1)

If issues found:
```
🔴 IMMEDIATE ACTIONS:
   1. Consolidate repeated results to ONE location
   2. Replace duplicates with references

🛠️ RECOMMENDED COMMANDS:
   [After fixing:]
   /check-content-redundancy   → Verify no duplicates
   /check-content-placement    → Check example/proof placement

   [When L1 passes:]
   /check-paper-consistency    → Move to L2
```

---

### After /check-content-placement (L1)

If issues found:
```
🔴 IMMEDIATE ACTIONS:
   1. Move assumptions to just-in-time locations
   2. Move examples near their theorems
   3. Move proofs to appendix if too long

🛠️ RECOMMENDED COMMANDS:
   [After fixing:]
   /check-content-placement    → Verify placement
   /check-paper-flow           → Check section transitions
```

---

### After /check-ms-style (L3)

If issues found:
```
🔴 IMMEDIATE ACTIONS:
   1. Add managerial insights to: [locations]
   2. Strengthen prescriptions in: [locations]
   3. Ensure service framing in: [locations]

🛠️ RECOMMENDED COMMANDS:
   [After fixing:]
   /check-ms-style             → Verify MS style

   [When L3 passes:]
   /polish-paper               → Move to L4 (language)
```

---

### After /check-paper-flow (L1)

If issues found:
```
🔴 IMMEDIATE ACTIONS:
   1. Fix transition at: [section boundaries]
   2. Add forward reference at: [locations]
   3. Improve coherence in: [sections]

🛠️ RECOMMENDED COMMANDS:
   [After fixing:]
   /check-paper-flow           → Verify flow
   /check-content-placement    → Related structure check
```

---

### After /polish-paper (L4)

```
🔴 IMMEDIATE ACTIONS:
   1. Review suggested edits
   2. Accept/reject each change
   3. Re-read for natural flow

🛠️ RECOMMENDED COMMANDS:
   /polish-paper               → Another pass if needed
   /paper-pipeline quick       → Final consistency check
   /paper-pipeline pre-submit MS → Submission checklist
```

---

## Decision Tree

```
Start: /paper-pipeline review MS
         │
         ▼
    ┌─────────────────┐
    │ Issues found?   │
    └────────┬────────┘
             │
     ┌───────┴───────┐
     │ YES           │ NO
     ▼               ▼
┌─────────────┐  ┌─────────────┐
│ Fix issues  │  │ Next level  │
│ manually    │  │ check       │
└──────┬──────┘  └──────┬──────┘
       │                │
       ▼                │
┌─────────────┐         │
│ Re-run same │         │
│ check       │◄────────┘
└──────┬──────┘
       │
       ▼
    Repeat until all levels pass
         │
         ▼
    /paper-pipeline pre-submit MS
```
