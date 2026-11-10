"""Charlson comorbidity mapping tables based on Quan (2005).

Reference:
    Quan H, Sundararajan V, Halfon P, et al.
    "Coding algorithms for defining comorbidities in ICD-9-CM and ICD-10
    administrative data."
    Medical Care. 2005;43(11):1130-1139.

Codes are stored as prefix lists (without dots). A code matches a category
if it starts with any prefix in the corresponding list.
"""

# ---------------------------------------------------------------------------
# Helper to generate string prefixes from numeric ranges
# ---------------------------------------------------------------------------

def _range_prefixes(start: int, end: int, width: int = 3) -> list[str]:
    return [str(i).zfill(width) for i in range(start, end + 1)]


def _icd10_range(letter: str, start: int, end: int) -> list[str]:
    return [f"{letter}{str(i).zfill(2)}" for i in range(start, end + 1)]


# ---------------------------------------------------------------------------
# 17 Charlson categories
# ---------------------------------------------------------------------------
CATEGORIES = [
    "mi", "chf", "pvd", "cevd", "dementia", "cpd", "rheumd", "pud",
    "mld", "diab", "diabwc", "hp", "rend", "canc", "msld", "metacanc", "aids",
]

# ---------------------------------------------------------------------------
# ICD-9-CM prefixes (Quan 2005, Table 1)
# ---------------------------------------------------------------------------
ICD9 = {
    "mi": ["410", "412"],
    "chf": [
        "39891", "40201", "40211", "40291", "40401", "40403",
        "40411", "40413", "40491", "40493",
        "4254", "4255", "4256", "4257", "4258", "4259", "428",
    ],
    "pvd": [
        "0930", "4373", "440", "441",
        "4431", "4432", "4433", "4434", "4435", "4436", "4437", "4438", "4439",
        "4471", "5571", "5579", "V434",
    ],
    "cevd": ["36234", "430", "431", "432", "433", "434", "435", "436", "437", "438"],
    "dementia": ["290", "2941", "3312"],
    "cpd": [
        "4168", "4169",
        *_range_prefixes(490, 496),
        *_range_prefixes(500, 505),
        "5064", "5081", "5088",
    ],
    "rheumd": [
        "4465", "7100", "7101", "7102", "7103", "7104",
        "7140", "7141", "7142", "7148", "725",
    ],
    "pud": ["531", "532", "533", "534"],
    "mld": [
        "07022", "07023", "07032", "07033", "07044", "07054",
        "0706", "0709", "570", "571",
        "5733", "5734", "5738", "5739", "V427",
    ],
    "diab": ["2500", "2501", "2502", "2503", "2508", "2509"],
    "diabwc": ["2504", "2505", "2506", "2507"],
    "hp": [
        "3341", "342", "343",
        "3440", "3441", "3442", "3443", "3444", "3445", "3446", "3449",
    ],
    "rend": [
        "40301", "40311", "40391", "40402", "40403", "40412", "40413",
        "40492", "40493", "582",
        "5830", "5831", "5832", "5833", "5834", "5835", "5836", "5837",
        "585", "586", "5880", "V420", "V451", "V56",
    ],
    "canc": [
        *_range_prefixes(140, 172),
        *_range_prefixes(174, 195),
        *_range_prefixes(200, 208),
        "2386",
    ],
    "msld": ["4560", "4561", "4562", "5722", "5723", "5724", "5728"],
    "metacanc": ["196", "197", "198", "199"],
    "aids": ["042", "043", "044"],
}

# ---------------------------------------------------------------------------
# ICD-10 prefixes (Quan 2005, Table 2)
# ---------------------------------------------------------------------------
ICD10 = {
    "mi": ["I21", "I22", "I252"],
    "chf": [
        "I099", "I110", "I130", "I132", "I255",
        "I420", "I425", "I426", "I427", "I428", "I429",
        "I43", "I50", "P290",
    ],
    "pvd": [
        "I70", "I71", "I731", "I738", "I739", "I771",
        "I790", "I792", "K551", "K558", "K559",
        "Z958", "Z959",
    ],
    "cevd": [
        "G45", "G46", "H340",
        *_icd10_range("I", 60, 69),
    ],
    "dementia": ["F00", "F01", "F02", "F03", "F051", "G30", "G311"],
    "cpd": [
        "I278", "I279",
        *_icd10_range("J", 40, 47),
        *_icd10_range("J", 60, 67),
        "J684", "J701", "J703",
    ],
    "rheumd": ["M05", "M06", "M315", "M32", "M33", "M34", "M351", "M353", "M360"],
    "pud": ["K25", "K26", "K27", "K28"],
    "mld": [
        "B18",
        "K700", "K701", "K702", "K703", "K709",
        "K713", "K714", "K715", "K717",
        "K73", "K74",
        "K760", "K762", "K763", "K764", "K768", "K769",
        "Z944",
    ],
    "diab": [
        "E100", "E101", "E106", "E108", "E109",
        "E110", "E111", "E116", "E118", "E119",
        "E120", "E121", "E126", "E128", "E129",
        "E130", "E131", "E136", "E138", "E139",
        "E140", "E141", "E146", "E148", "E149",
        # ICD-10-CM additions for MIMIC-IV (E08/E09 = secondary diabetes)
        "E080", "E081", "E086", "E088", "E089",
        "E090", "E091", "E096", "E098", "E099",
    ],
    "diabwc": [
        "E102", "E103", "E104", "E105", "E107",
        "E112", "E113", "E114", "E115", "E117",
        "E122", "E123", "E124", "E125", "E127",
        "E132", "E133", "E134", "E135", "E137",
        "E142", "E143", "E144", "E145", "E147",
        # ICD-10-CM additions for MIMIC-IV (E08/E09 with complications)
        "E082", "E083", "E084", "E085", "E087",
        "E092", "E093", "E094", "E095", "E097",
    ],
    "hp": [
        "G041", "G114", "G801", "G802",
        "G81", "G82",
        "G830", "G831", "G832", "G833", "G834", "G839",
    ],
    "rend": [
        "I120", "I131",
        "I132",  # ICD-10-CM: hypertensive heart+renal disease with HF+CKD
        "N032", "N033", "N034", "N035", "N036", "N037",
        "N052", "N053", "N054", "N055", "N056", "N057",
        "N18", "N19", "N250",
        "Z490", "Z491", "Z492", "Z940", "Z992",
    ],
    "canc": [
        *_icd10_range("C", 0, 26),
        *_icd10_range("C", 30, 34),
        *_icd10_range("C", 37, 41),
        "C43",
        *_icd10_range("C", 45, 58),
        *_icd10_range("C", 60, 76),
        *_icd10_range("C", 81, 85),
        "C88",
        *_icd10_range("C", 90, 97),
        "C4A",  # ICD-10-CM: Merkel cell carcinoma
        "C7A",  # ICD-10-CM: malignant neuroendocrine tumors
    ],
    "msld": [
        "I850", "I859", "I864", "I982",
        "K704", "K711", "K721", "K729",
        "K765", "K766", "K767",
    ],
    "metacanc": [
        "C77", "C78", "C79", "C80",
        "C7B",  # ICD-10-CM: secondary neuroendocrine tumors
    ],
    "aids": ["B20", "B21", "B22", "B24"],
}

# ---------------------------------------------------------------------------
# Charlson 1987 weights
# ---------------------------------------------------------------------------
WEIGHTS = {
    "mi": 1, "chf": 1, "pvd": 1, "cevd": 1, "dementia": 1,
    "cpd": 1, "rheumd": 1, "pud": 1, "mld": 1, "diab": 1,
    "diabwc": 2, "hp": 2, "rend": 2, "canc": 2,
    "msld": 3, "metacanc": 6, "aids": 6,
}

# Hierarchy: higher-weight category suppresses lower-weight one
HIERARCHY = [
    ("diabwc", "diab"),    # diabwc (2) suppresses diab (1)
    ("msld", "mld"),       # msld (3) suppresses mld (1)
    ("metacanc", "canc"),  # metacanc (6) suppresses canc (2)
]


def match_code(code: str, version: int) -> list[str]:
    """Return all Charlson categories matched by a single ICD code."""
    table = ICD9 if version == 9 else ICD10
    code = code.strip().upper()
    matched = []
    for cat, prefixes in table.items():
        for prefix in prefixes:
            if code.startswith(prefix.upper()):
                matched.append(cat)
                break
    return matched


# Pre-computed prefix tuples for fast vectorized matching with str.startswith
_PREFIX_TUPLES_9 = {cat: tuple(p.upper() for p in prefixes) for cat, prefixes in ICD9.items()}
_PREFIX_TUPLES_10 = {cat: tuple(p.upper() for p in prefixes) for cat, prefixes in ICD10.items()}


def vectorized_match(codes: "pd.Series", versions: "pd.Series") -> "pd.DataFrame":
    """Vectorized Charlson matching for a DataFrame of ICD codes.

    Args:
        codes: Series of ICD code strings.
        versions: Series of ICD version integers (9 or 10).

    Returns:
        DataFrame with one bool column per Charlson category, indexed like codes.
    """
    import pandas as pd  # noqa: F811

    codes_upper = codes.str.strip().str.upper()
    is_v9 = versions == 9
    is_v10 = ~is_v9

    result = pd.DataFrame(False, index=codes.index, columns=CATEGORIES)

    for cat in CATEGORIES:
        mask = pd.Series(False, index=codes.index)
        # ICD-9
        if is_v9.any():
            prefixes_9 = _PREFIX_TUPLES_9[cat]
            if prefixes_9:
                mask |= is_v9 & codes_upper.str.startswith(prefixes_9)
        # ICD-10
        if is_v10.any():
            prefixes_10 = _PREFIX_TUPLES_10[cat]
            if prefixes_10:
                mask |= is_v10 & codes_upper.str.startswith(prefixes_10)
        result[cat] = mask

    return result


def charlson_score(flags: dict[str, int], age: int | None = None) -> tuple[int, int | None]:
    """Compute Charlson score from binary category flags.

    Applies hierarchy (diabwc>diab, msld>mld, metacanc>canc).

    Returns:
        (charlson_icd, charlson_icd_age) where charlson_icd_age is None
        if age is None.
    """
    h = dict(flags)
    for dominant, suppressed in HIERARCHY:
        if h.get(dominant, 0) == 1:
            h[suppressed] = 0

    score = sum(h.get(cat, 0) * w for cat, w in WEIGHTS.items())

    age_score = None
    if age is not None:
        age_points = 0
        if age >= 50:
            age_points = min((age - 40) // 10, 4)
        age_score = score + age_points

    return score, age_score


def vectorized_score(labels: "pd.DataFrame", ages: "pd.Series | None" = None) -> "pd.DataFrame":
    """Vectorized Charlson score computation.

    Args:
        labels: DataFrame with 17 binary Charlson columns.
        ages: Optional Series of ages, aligned with labels index.

    Returns:
        DataFrame with charlson_icd and charlson_icd_age columns.
    """
    import pandas as pd  # noqa: F811
    import numpy as np

    df = labels[CATEGORIES].copy()

    # Apply hierarchy
    for dominant, suppressed in HIERARCHY:
        df.loc[df[dominant] == 1, suppressed] = 0

    # Weighted sum
    weight_arr = np.array([WEIGHTS[cat] for cat in CATEGORIES])
    scores = df[CATEGORIES].values @ weight_arr

    result = pd.DataFrame({"charlson_icd": scores}, index=labels.index)

    if ages is not None:
        age_points = np.clip((ages.values - 40) // 10, 0, 4)
        age_points = np.where(ages.values >= 50, age_points, 0)
        result["charlson_icd_age"] = scores + age_points
    else:
        result["charlson_icd_age"] = pd.NA

    return result
