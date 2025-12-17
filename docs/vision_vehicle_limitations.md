# Vision-Based Vehicle Extraction: Limitations and Design Decisions

## 1. Executive Summary

Vision-based vehicle extraction works well for **explicitly labeled fields** in structured layouts (tables, forms) and can **infer semantic fields from cohesive natural language** (e.g., notes sentences). However, it has fundamental limitations that cannot be fixed through prompt engineering alone.

**What works:**
- Extracting labeled fields from tables (VIN, year, make, model, color, mileage)
- Inferring semantic fields (`body_style`, `fuel_type`, `transmission`) from natural language sentences
- Preserving invalid values for warning generation (year=1899, mileage=-100, bad email)

**What doesn't work:**
- Inferring semantic fields from table columns (when not explicitly labeled)
- Inferring fields from scattered text across separate blocks
- Extracting email addresses from table cells
- Correcting VIN character recognition errors in handwritten documents

**Current test failures:**
- PDF vehicle documents: Row 6 missing fields (extraction failure)
- Handwritten PDF/image: Semantic fields (`body_style`, `fuel_type`, `transmission`) consistently `null`
- PNG images: Semantic fields missing when data is in table format (but work when in notes)
- All Vision sources: `owner_email` consistently `null` when not in natural language context

**Root cause:** Vision's inference boundary is model-limited, not prompt-limited. The model performs local semantic inference (within sentences/paragraphs) but does not perform cross-field inference (across table columns or separate text blocks).

**Next steps:** Architectural decision required: add post-Vision inference logic to extract semantic fields from `notes` and email addresses from all fields, or update test expectations to accept `null` values when fields are not explicitly labeled.

---

## 2. Vision Extraction Pipeline Overview

### Flow
1. **Source routing:** PDFs/images → `_extract_table_with_vision_api()`
2. **Image preparation:** PDFs converted to images; each page processed separately
3. **Vision API call:** System prompt (inference rules) + user prompt (image data)
4. **Output processing:** Parse JSON, normalize types, filter invalid VINs, add `_is_vision_extracted` flag
5. **Conversion:** 2D array → row dictionaries via `rows2d_to_objects()`
6. **Normalization:** `normalize_v2()` applies mappings, preserves Vision-extracted fields, generates warnings
7. **Output:** Normalized rows with warnings

### Key Design Decisions
- **Invalid values preserved:** `_normalize_vision_value()` preserves invalid values (year=1899, mileage=-100) for warning generation
- **VIN correction disabled:** `correct_vin_once()` skipped for Vision-extracted rows to preserve exact Vision output
- **Field promotion disabled:** `promote_fields_from_notes()` skipped for Vision-extracted rows (Vision should infer directly)
- **Vision safeguard:** All canonical fields from Vision-extracted rows are preserved in `normalize_v2()`, including `null` values

---

## 3. What Vision Can Do vs Cannot Do

### Vision CAN Do:
- ✅ **Perception:** Read text from images with high accuracy for typed text
- ✅ **Local semantic inference:** Infer fields from cohesive natural language
  - Example: "2024 Toyota Camry sedan painted blue. 12,345 miles. Fuel: gasoline. 8-speed automatic transmission."
  - Result: All fields inferred correctly (`body_style=sedan`, `fuel_type=gasoline`, `transmission=automatic`)
- ✅ **Structured extraction:** Extract labeled fields from tables/forms
  - Example: Table with columns "VIN", "Year", "Make", "Model"
  - Result: All labeled columns extracted correctly

### Vision CANNOT Do:
- ❌ **Cross-field inference:** Infer semantic fields from table structure
  - Example: Table with columns "VIN", "Year", "Make", "Model" (no `body_style` column)
  - Result: `body_style=null` even if visible in notes or other columns
- ❌ **Cross-block inference:** Infer fields from scattered text across separate blocks
  - Example: VIN in one block, "sedan" mentioned in another block
  - Result: `body_style=null` (Vision treats blocks independently)
- ❌ **Email extraction from tables:** Extract email addresses from table cells
  - Example: Table cell containing "john.doe@example.com"
  - Result: `owner_email=null` (Vision only extracts emails from natural language context)
- ❌ **VIN correction:** Correct OCR character recognition errors
  - Example: Handwritten "O" misread as "0", "I" misread as "1"
  - Result: Corrupted VINs preserved (correction disabled for Vision-extracted rows)

### Why This Matters:
Vision's inference boundary is **model-limited**, not **prompt-limited**. Adding more prompt instructions cannot force the model to perform cross-field or cross-block inference. The model's architecture determines what types of inference are possible.

---

## 4. Source-by-Source Behavior

### A. PDF Vehicle Documents (Typed PDFs)

**Status:** Partial success (rows 1-5 pass, row 6 fails)

**What works:**
- Rows 1-5: All fields extracted correctly via OCR path
- Invalid values preserved: `year=1899`, `mileage=-100`, `bad-email-format` → warnings generated correctly

**What fails:**
- Row 6: All fields `null` when Vision path is triggered (OCR path extracts correctly, but Vision path fails)
- Semantic fields: `body_style`, `fuel_type`, `transmission` often `null` even when visible

**Root cause:** Row 6 appears on a different page or in a format Vision doesn't recognize. OCR path extracts it, but Vision path (when triggered) fails.

---

### B. Handwritten PDF Vehicle Documents

**Status:** Partial success (basic fields work, semantic fields fail)

**What works:**
- Basic fields: `vin` (with corruption), `year`, `make`, `model`, `color`, `mileage`
- Invalid values preserved: Negative mileage → warnings generated

**What fails:**
- VIN corruption: Character substitutions (G→4, F→M, Z→7, S→5, etc.)
- Semantic fields: `body_style`, `fuel_type`, `transmission`, `owner_email` consistently `null`
- Example: Notes contain "sedan", "gasoline", "automatic" → Vision returns `null` for all three

**Root cause:** 
- VIN corruption: OCR limitation (character misrecognition)
- Semantic fields: Vision infers from cohesive notes but not from scattered handwritten text

---

### C. PNG Image Vehicle Documents (Typed Images)

**Status:** Conditional success (works for notes, fails for tables)

**What works:**
- When data is in a single notes sentence: All fields extracted correctly (including semantic fields)
- Example: "2024 Toyota Camry sedan painted blue. 12,345 miles. Fuel: gasoline. 8-speed automatic transmission."
- Result: `body_style=sedan`, `fuel_type=gasoline`, `transmission=automatic` ✅

**What fails:**
- When data is in table columns: Semantic fields `null`
- Example: Table with columns "VIN", "Year", "Make", "Model" (no `body_style` column)
- Result: `body_style=null`, `fuel_type=null`, `transmission=null` ❌
- `owner_email`: Consistently `null` even when visible in table cells

**Root cause:** Vision infers from natural language in cohesive text but not from table structure.

---

### D. Handwritten Image Vehicle Documents

**Status:** Worst performance (basic fields work, semantic fields never work)

**What works:**
- Basic fields: `vin` (corrupted), `year`, `make`, `model`, `color`, `mileage`

**What fails:**
- VIN corruption: Higher than handwritten PDFs (no OCR fallback)
- Semantic fields: `body_style`, `fuel_type`, `transmission`, `owner_email` always `null`
- Warnings: Incomplete (corrupted VINs not flagged)

**Root cause:** Vision-only path (no OCR fallback) + handwritten text = lower accuracy + no semantic inference.

---

## 5. VIN Corruption Explanation

### What Happens:
Handwritten VINs are misread by OCR, causing character substitutions:
- `HBUSRJGF4CBFPR9BN` → `HBURSJ4MFBCFPR9BN` (G→4, F→M)
- `3R5UAL4YUKPYGF1GZ` → `3R5UAL4YUKPYGF1G7` (Z→7)
- `ST420RJ98FDHKL4E` → `5T4Z0RJ98FDHKL4HE` (S→5, 4→Z, 2→0, F→H, E→HE)

### Why It Happens:
- OCR limitation: Character misrecognition (0/O, 1/I, 4/A, etc.)
- Not fixable by prompts: This is a perception issue, not an inference issue

### Current Behavior:
- VIN correction disabled for Vision-extracted rows (to preserve exact Vision output)
- No validation warnings for corrupted VINs
- Corrupted VINs pass through to final output

### Options:
1. **Accept with warnings:** Add VIN validation logic to flag corrupted VINs (character substitutions, invalid format)
2. **Enable correction:** Re-enable `correct_vin_once()` for Vision-extracted rows (may introduce false corrections)
3. **Specialized OCR:** Use specialized OCR for handwritten text (may improve accuracy but not eliminate errors)

---

## 6. Why Prompt Changes Plateaued

### What We Tried:
1. **Added CRITICAL EXTRACTION RULES:** Explicitly required inference for all canonical fields
2. **Removed conflicting instructions:** Removed "Do NOT infer missing values" that contradicted inference requirements
3. **Added inference examples:** Provided specific examples of how to infer fields from natural language

### What Happened:
- **Structural improvement:** Prompts now explicitly require inference
- **Behavioral stagnation:** Vision still returns `null` for semantic fields in tables/scattered text

### Why It Didn't Work:
Vision's inference boundary is **model-limited**, not **prompt-limited**:
- Vision performs **local semantic inference** (within sentences/paragraphs) ✅
- Vision does NOT perform **cross-field inference** (across table columns or separate text blocks) ❌

**Example:**
- Cohesive notes: "2024 Toyota Camry sedan painted blue. 12,345 miles. Fuel: gasoline. 8-speed automatic transmission."
  - Result: All fields inferred correctly ✅
- Scattered text: VIN in one block, "sedan" mentioned in another block
  - Result: `body_style=null` (Vision treats blocks independently) ❌

**Implication:** Prompts can guide local inference but cannot force cross-field inference. The model's architecture determines what types of inference are possible.

---

## 7. Design Options Moving Forward

### Option A: Post-Vision Inference Logic

**What:** Add inference logic in `normalize_v2()` to extract semantic fields from `notes` and email addresses from all fields for Vision-extracted rows.

**Implementation:**
- Extract `body_style`, `fuel_type`, `transmission` from `notes` field using regex/keyword matching
- Extract `owner_email` from all fields using regex pattern matching
- Apply only when field is `null` (don't override explicit Vision extraction)

**Pros:**
- Improves extraction coverage without changing Vision prompts
- Works for all Vision sources (PDF, handwritten PDF, PNG, handwritten image)
- Maintains current test expectations

**Cons:**
- Adds complexity to `normalize_v2()`
- May introduce false positives (incorrect inference)
- Requires maintenance of regex patterns

**Effort:** Medium (2-3 days)

---

### Option B: Update Test Expectations

**What:** Update test truth files to accept `null` values for semantic fields when not explicitly labeled.

**Implementation:**
- Update `pdf_vehicle_documents.expected.json` to set `body_style=null`, `fuel_type=null`, `transmission=null` for rows where fields are not explicitly labeled
- Update `handwritten_vehicles.json` to set semantic fields to `null` (current behavior)
- Update `vehicles_png.expected.json` to set semantic fields to `null` for table-based rows

**Pros:**
- No code changes required
- Aligns tests with actual Vision capabilities
- Clearer expectations for future development

**Cons:**
- Reduces test coverage (fewer fields validated)
- May mask real extraction issues
- Doesn't improve extraction quality

**Effort:** Low (1 day)

---

### Option C: Hybrid Approach

**What:** Combine Option A and Option B:
- Add post-Vision inference logic for `notes`-based extraction
- Update test expectations to accept `null` for table-based rows where inference is not possible

**Implementation:**
- Extract semantic fields from `notes` when available (post-Vision inference)
- Accept `null` for semantic fields when not in `notes` and not explicitly labeled (updated test expectations)

**Pros:**
- Maximizes extraction coverage where possible
- Realistic expectations for table-based rows
- Balanced approach

**Cons:**
- Requires both code changes and test updates
- More complex to maintain

**Effort:** Medium-High (3-4 days)

---

### Recommendation: Option C (Hybrid Approach)

**Rationale:**
- Post-Vision inference improves extraction for notes-based sources (handwritten PDF, PNG with notes)
- Updated test expectations align with Vision's actual capabilities for table-based sources
- Balanced approach that maximizes coverage while maintaining realistic expectations

---

## 8. Open Questions for Team Review

### 1. VIN Corruption Handling
**Question:** How should we handle corrupted VINs in handwritten sources?

**Options:**
- A) Accept corrupted VINs with validation warnings (current behavior)
- B) Re-enable VIN correction for Vision-extracted rows (may introduce false corrections)
- C) Use specialized OCR for handwritten text (may improve accuracy but not eliminate errors)

**Recommendation:** Option A (accept with warnings) - VIN correction is risky and may introduce false corrections.

---

### 2. Semantic Field Inference Strategy
**Question:** Should we add post-Vision inference logic to extract semantic fields from `notes`?

**Options:**
- A) Yes, add post-Vision inference for all Vision-extracted rows
- B) No, accept `null` values and update test expectations
- C) Hybrid: Add inference for notes-based sources, accept `null` for table-based sources

**Recommendation:** Option C (hybrid) - Maximizes coverage where possible while maintaining realistic expectations.

---

### 3. Email Extraction Strategy
**Question:** Should we add regex-based email extraction to scan all fields for email addresses?

**Options:**
- A) Yes, add email extraction logic in `normalize_v2()`
- B) No, accept `null` values and update test expectations
- C) Only for Vision-extracted rows (don't change behavior for other sources)

**Recommendation:** Option A (add email extraction) - Email addresses are easy to detect with regex and improve extraction coverage.

---

### 4. Test Expectation Alignment
**Question:** Should we update test expectations to accept `null` for semantic fields when not explicitly labeled?

**Options:**
- A) Yes, update all test truth files to align with Vision capabilities
- B) No, keep current expectations and add post-Vision inference logic
- C) Partial: Update expectations for table-based rows, keep current expectations for notes-based rows

**Recommendation:** Option C (partial) - Aligns with Vision's actual capabilities while maintaining high expectations for notes-based sources.

---

### 5. Row 6 Extraction Failure
**Question:** How should we handle the Row 6 extraction failure in typed PDFs?

**Options:**
- A) Investigate why Vision path fails for Row 6 and fix the root cause
- B) Ensure OCR path is always attempted before Vision path (current behavior should work)
- C) Accept partial extraction and update test expectations

**Recommendation:** Option B (ensure OCR path is always attempted) - OCR path extracts Row 6 correctly; Vision path should only be fallback.

---

### 6. Warning Generation for Corrupted VINs
**Question:** Should we add VIN validation warnings for corrupted VINs?

**Options:**
- A) Yes, add VIN validation logic to flag character substitutions and invalid formats
- B) No, corrupted VINs are acceptable (current behavior)
- C) Only for Vision-extracted rows (don't change behavior for other sources)

**Recommendation:** Option A (add VIN validation warnings) - Helps identify data quality issues without changing extraction behavior.

---

## Appendix: Field Extraction Status by Source

| Field | PDF | Handwritten PDF | PNG Image | Handwritten Image | Notes |
|-------|-----|-----------------|-----------|-------------------|-------|
| `vin` | ✅ | ⚠️ (corrupted) | ✅ | ⚠️ (corrupted) | OCR limitation for handwritten |
| `year` | ✅ | ✅ | ✅ | ✅ | Reliable |
| `make` | ✅ | ✅ | ✅ | ✅ | Reliable |
| `model` | ✅ | ✅ | ✅ | ✅ | Reliable |
| `color` | ✅ | ✅ | ✅ | ✅ | Reliable |
| `mileage` | ✅ | ✅ | ✅ | ✅ | Reliable |
| `body_style` | ❌ (null) | ❌ (null) | ⚠️ (notes only) | ❌ (null) | Requires inference |
| `fuel_type` | ❌ (null) | ❌ (null) | ⚠️ (notes only) | ❌ (null) | Requires inference |
| `transmission` | ❌ (null) | ❌ (null) | ⚠️ (notes only) | ❌ (null) | Requires inference |
| `owner_email` | ❌ (null) | ❌ (null) | ❌ (null) | ❌ (null) | Requires inference |

**Legend:**
- ✅ = Consistently extracted correctly
- ⚠️ = Partially extracted (works in some contexts, fails in others)
- ❌ = Consistently `null` or missing

---

## Conclusion

Vision-based vehicle extraction works well for explicitly labeled fields and can infer semantic fields from cohesive natural language. However, it has fundamental limitations that cannot be fixed through prompt engineering alone. The model's inference boundary is model-limited, not prompt-limited.

**Recommended next steps:**
1. Add post-Vision inference logic to extract semantic fields from `notes` and email addresses from all fields
2. Update test expectations to accept `null` for semantic fields in table-based rows where inference is not possible
3. Add VIN validation warnings for corrupted VINs
4. Ensure OCR path is always attempted before Vision path for typed PDFs

This hybrid approach maximizes extraction coverage where possible while maintaining realistic expectations aligned with Vision's actual capabilities.
