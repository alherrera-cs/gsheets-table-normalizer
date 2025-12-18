# Vehicle Extraction System: Code and Prompt Audit

**Date:** 2024  
**Purpose:** Identify contradictions, redundancies, and areas for improvement in vehicle extraction code and AI prompts

---

## 🔴 CRITICAL CONTRADICTIONS

### 1. **Normalization Contradiction (MAJOR)**

**Location:** `sources.py` lines 4060 vs 4777-4782

**Contradiction:**
- **Line 4060 (System Prompt):** `"Do NOT normalize or convert values (preserve original format)"`
- **Lines 4777-4782 (User Prompt):** `"NORMALIZATION REQUIREMENTS: Normalize inferred values to canonical forms..."`

**Impact:** Vision API receives conflicting instructions - told to preserve raw values in system prompt, but told to normalize in user prompt. This causes inconsistent behavior (e.g., "gas" not normalized to "gasoline").

**Fix Required:** Remove normalization instructions from Vision prompts. All normalization should happen in `normalize_v2()`.

---

### 2. **Inference Scope Contradiction**

**Location:** `sources.py` lines 3940-3961 vs 4058-4060

**Contradiction:**
- **Lines 3940-3961:** `"You must attempt to extract EVERY canonical vehicle field... even if the value must be inferred from natural language"`
- **Lines 4058-4060:** `"Extract values EXACTLY as written... Do NOT reword, summarize, paraphrase"`

**Impact:** Unclear whether Vision should infer semantic fields or only extract verbatim text.

**Fix Required:** Clarify: Infer semantic fields (body_style, fuel_type, transmission, mileage) when not explicitly labeled. Extract verbatim for explicitly labeled fields.

---

## 🔄 REDUNDANCY ISSUES

### 3. **Duplicate Prompt Sections**

**Location:** `sources.py` lines 3936-4080 vs 4093-4226

**Issue:** Two nearly identical system prompts (main and fallback) with ~290 lines duplicated.

**Fix Required:** Extract common sections into reusable function.

---

### 4. **Duplicate Extraction Rules**

**Location:** Multiple locations

**Issue:** "EXTRACTION RULES (CRITICAL)" appears 3 times:
- Line 4057 (main prompt)
- Line 4207 (fallback prompt)  
- Line 4756 (user prompt)

**Fix Required:** Define once, reference elsewhere.

---

### 5. **Duplicate Section-Based Extraction Instructions**

**Location:** `sources.py` lines 3988-4007, 4142-4157, 4660-4675

**Issue:** "SECTION-BASED SEPARATION" appears 3 times with slight variations (~20 lines each).

**Fix Required:** Consolidate into single section.

---

### 6. **Duplicate IMAGE TYPE DETECTION**

**Location:** `sources.py` lines 4011-4046 vs 4161-4196

**Issue:** "IMAGE TYPE DETECTION" duplicated in main and fallback prompts (~35 lines each).

**Fix Required:** Extract to function.

---

### 7. **Duplicate VIN Validation Rules**

**Location:** Multiple locations

**Issue:** VIN validation rules repeated in system prompt, user prompt, and code comments.

**Fix Required:** Define once, reference elsewhere.

---

## ⚠️ CODE LOGIC CONTRADICTIONS

### 8. **Preserve vs Normalize in `_normalize_vision_value`**

**Location:** `sources.py` lines 584-723

**Contradiction:**
- `preserve_raw=True` for PDFs: "preserve exact values"
- But still normalizes: fuel_type (line 701-708), transmission (line 690-699), body_style (line 710-712)

**Impact:** PDF values are normalized despite "preserve_raw" intent.

**Fix Required:** If preserving raw, skip normalization. If normalizing, remove "preserve_raw" logic.

---

### 9. **Notes Field: Verbatim vs Inference**

**Location:** `sources.py` lines 4070-4076, 4219-4225, 4653

**Contradiction:**
- "Copy VERBATIM - word-for-word, character-for-character"
- But also: "Infer from natural language when possible"

**Impact:** Unclear whether notes should be verbatim or inferred.

**Fix Required:** Clarify: Notes are verbatim. Other fields can be inferred.

---

## 🔁 REDUNDANT CODE PATTERNS

### 10. **Duplicate Header Token Filtering**

**Location:** Multiple locations

**Issue:** `invalid_vin_tokens` defined in 6+ places:
- Line 1445 (extract_fields_from_block)
- Line 4576 (OCR path)
- Line 4880 (Vision API)
- Line 5292 (IMAGE aggregation)
- Line 5352 (IMAGE aggregation)
- Line 6326 (table extraction)

**Fix Required:** Define once as constant, import everywhere.

---

### 11. **Duplicate Document-Level Defaults Extraction**

**Location:** Multiple locations

**Issue:** `_extract_document_level_defaults` called in 4+ places with same logic.

**Fix Required:** Already a function - ensure it's only called once per document.

---

### 12. **Duplicate Field List Definitions**

**Location:** Multiple locations

**Issue:** 
- Critical fields: `['year', 'make', 'model']` appears in 3+ places
- Semantic fields: `['body_style', 'fuel_type', 'transmission', 'mileage']` appears in 4+ places

**Fix Required:** Define as constants in `schema.py`.

---

## 📋 PROMPT CLARITY ISSUES

### 13. **Conflicting Null vs Omit Instructions**

**Location:** Multiple locations

**Issue:** Same instruction repeated 3+ times: "return null — NEVER omit the field"

**Impact:** Redundant but consistent.

**Fix Required:** Consolidate into single clear instruction.

---

### 14. **Overlapping Inference Examples**

**Location:** Lines 3947-3952, 4104-4109, 4764-4775

**Issue:** Same inference examples repeated 3 times (sedan → body_style, gas → fuel_type, etc.)

**Fix Required:** Define once, reference elsewhere.

---

## 🎯 PRIORITY FIXES

### High Priority (Breaking Contradictions)
1. ✅ **Remove normalization from Vision prompts** (lines 4060, 4777-4782)
2. ✅ **Clarify inference scope** (when to infer vs verbatim)
3. ✅ **Fix preserve_raw logic** (ensure PDFs preserve raw, normalization in normalize_v2)

### Medium Priority (Code Quality)
4. Consolidate duplicate prompts (main vs fallback)
5. Extract header token lists to constants
6. Consolidate section-based extraction instructions

### Low Priority (Cleanup)
7. Consolidate duplicate field lists
8. Remove redundant null/omit instructions
9. Consolidate inference examples

---

## 📝 NOTES

- All normalization should happen in `normalize_v2()` via transforms
- Vision API should extract raw values only (verbatim or inferred, but not normalized)
- This separation ensures consistent behavior across all sources
