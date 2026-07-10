"""
data.py — CynLr team tensors from embedded snapshot.
Returns numpy arrays ready for DRVEngine.
"""
import numpy as np

CAP_BY_LEVEL = {
    "Principal Engineer": 150.0,
    "Senior Engineer":    120.0,
    "Lead":               100.0,
    "Engineer 3":          85.0,
    "Engineer 2":          70.0,
    "Engineer 1":          55.0,
    "Engineer Intern":     45.0,
    "Intern":              35.0,
}

# (key, strategic_weight)
PROJECTS = [
    ("Audi(Door Assembly)",              2.0),
    ("Lam Research (QC Cup Assembly)",   2.0),
    ("AMAT -1 (Kitting)",               2.0),
    ("AMAT -2 (Frame Handling)",        2.0),
    ("Schneider Electric - India",       2.0),
    ("Hyundai",                         2.0),
    ("CyRo",                            1.8),
    ("CyNoid",                          1.8),
    ("CLX",                             1.6),
    ("Robotics Research",               1.2),
    ("SW Tools",                        1.0),
    ("HW Tools",                        1.0),
    ("Gripper",                         1.0),
    ("DevOps",                          0.8),
    ("Code Refactoring",                0.8),
    ("Demo Development",                0.8),
    ("Research/Exploration - HW",       0.6),
    ("Research/Exploration - SW",       0.6),
    ("Neuroscience, Vision Research IISC", 0.6),
    ("Product Packaging",               0.5),
    ("Documentation",                   0.4),
    ("Hiring",                          0.4),
]

PEOPLE = [
    {"name": "Aashlesh Anilkumar Pardhi",      "level": "Intern",            "allocs": {"CLX": 50, "Research/Exploration - HW": 25, "Product Packaging": 25}},
    {"name": "Adithi Shetty N",               "level": "Intern",            "allocs": {"CLX": 50, "Lam Research (QC Cup Assembly)": 50}},
    {"name": "A Kathir",                       "level": "Intern",            "allocs": {"Audi(Door Assembly)": 80, "SW Tools": 20}},
    {"name": "Ananthu V",                      "level": "Intern",            "allocs": {"CyRo": 100}},
    {"name": "Aryan Mathure",                  "level": "Intern",            "allocs": {"CyRo": 30, "Lam Research (QC Cup Assembly)": 70}},
    {"name": "Darshan M",                      "level": "Intern",            "allocs": {"Lam Research (QC Cup Assembly)": 50, "Gripper": 50}},
    {"name": "Dhayadharsh S M",               "level": "Intern",            "allocs": {"CLX": 90, "HW Tools": 10}},
    {"name": "G Pranav",                       "level": "Intern",            "allocs": {"Audi(Door Assembly)": 20, "SW Tools": 70, "Neuroscience, Vision Research IISC": 10}},
    {"name": "HARISANKAR A G",                "level": "Intern",            "allocs": {}},
    {"name": "HIMANSHU RAI",                  "level": "Intern",            "allocs": {"CLX": 40, "HW Tools": 60}},
    {"name": "Kavin Prakash M",               "level": "Intern",            "allocs": {}},
    {"name": "Mrithika Basker",               "level": "Intern",            "allocs": {"SW Tools": 100}},
    {"name": "Rishab Bohra",                  "level": "Intern",            "allocs": {"AMAT -1 (Kitting)": 20, "SW Tools": 10, "Neuroscience, Vision Research IISC": 70}},
    {"name": "Shalank N Kulkarni",            "level": "Intern",            "allocs": {"CyRo": 25, "CyNoid": 50, "Product Packaging": 25}},
    {"name": "Sharif S",                      "level": "Intern",            "allocs": {}},
    {"name": "Sri Abhishek Kulkarni",         "level": "Intern",            "allocs": {"Audi(Door Assembly)": 10, "AMAT -1 (Kitting)": 30, "Neuroscience, Vision Research IISC": 60}},
    {"name": "Vivaan Vipin Jain",             "level": "Intern",            "allocs": {"CLX": 10, "SW Tools": 40, "Gripper": 50}},
    {"name": "Krishnapranav Balasubramanian", "level": "Intern",            "allocs": {}},
    {"name": "K Vinoth Kumar",                "level": "Intern",            "allocs": {"Research/Exploration - HW": 10}},
    {"name": "A Y Ellakiya",                  "level": "Intern",            "allocs": {"Documentation": 100}},
    {"name": "Ajayprabhakar Kavungal",        "level": "Lead",              "allocs": {}},
    {"name": "Asmita Rajesh Pai",             "level": "Engineer 1",        "allocs": {}},
    {"name": "Bharath Jain",                  "level": "Engineer 2",        "allocs": {"HW Tools": 25, "Gripper": 25, "Product Packaging": 50}},
    {"name": "Chethan K",                     "level": "Engineer 2",        "allocs": {"CLX": 15, "CyRo": 10, "CyNoid": 40, "Research/Exploration - HW": 15, "Gripper": 10, "Hiring": 10}},
    {"name": "Deepti Bhat",                   "level": "Engineer 1",        "allocs": {"CyNoid": 60, "SW Tools": 10, "Code Refactoring": 20, "Hiring": 10}},
    {"name": "Edwin Justin",                  "level": "Engineer 1",        "allocs": {"CyRo": 25, "CyNoid": 50, "Audi(Door Assembly)": 25}},
    {"name": "Gagan Goutham N",               "level": "Engineer 2",        "allocs": {"CyNoid": 10, "AMAT -1 (Kitting)": 10, "Neuroscience, Vision Research IISC": 80}},
    {"name": "Goutham K",                     "level": "Engineer 1",        "allocs": {"CLX": 75, "HW Tools": 25}},
    {"name": "Gurubaran Elango",              "level": "Lead",              "allocs": {}},
    {"name": "Javeed Ahmad",                  "level": "Engineer 1",        "allocs": {"Lam Research (QC Cup Assembly)": 60, "AMAT -1 (Kitting)": 40}},
    {"name": "Kishika Singh",                 "level": "Engineer 1",        "allocs": {"Audi(Door Assembly)": 80, "Demo Development": 20}},
    {"name": "Lakshith Kodihalli Shashikumar","level": "Engineer 2",        "allocs": {"CyRo": 25, "CyNoid": 50, "Hiring": 25}},
    {"name": "Maha Zakir Khan",               "level": "Engineer 2",        "allocs": {}},
    {"name": "Manvanth G S",                  "level": "Engineer 2",        "allocs": {"SW Tools": 30, "Code Refactoring": 20, "DevOps": 40, "Hiring": 10}},
    {"name": "Mehul Amit Avalaskar",          "level": "Engineer 1",        "allocs": {"Audi(Door Assembly)": 25, "Lam Research (QC Cup Assembly)": 25, "Schneider Electric - India": 25, "Demo Development": 25}},
    {"name": "Mohit Jani",                    "level": "Engineer 2",        "allocs": {"Audi(Door Assembly)": 30, "Lam Research (QC Cup Assembly)": 50, "Robotics Research": 20}},
    {"name": "Mothukuri Jaswanth Venkat",     "level": "Engineer 2",        "allocs": {"SW Tools": 30, "Demo Development": 10, "Code Refactoring": 20, "DevOps": 30, "Hiring": 10}},
    {"name": "Narendhiran K S",               "level": "Engineer 2",        "allocs": {}},
    {"name": "Prakash Pulak Ghosh",           "level": "Engineer 3",        "allocs": {"Lam Research (QC Cup Assembly)": 80, "Schneider Electric - India": 5, "Research/Exploration - HW": 15}},
    {"name": "Pranav J Nair",                 "level": "Engineer 2",        "allocs": {}},
    {"name": "Sanjith Sudhakar",              "level": "Engineer 2",        "allocs": {"Audi(Door Assembly)": 100}},
    {"name": "Sankarshan D H",                "level": "Engineer 1",        "allocs": {"Lam Research (QC Cup Assembly)": 50, "Neuroscience, Vision Research IISC": 50}},
    {"name": "Satwik Agarwal",                "level": "Engineer 1",        "allocs": {"CLX": 30, "SW Tools": 20, "Demo Development": 20, "Code Refactoring": 20, "Hiring": 10}},
    {"name": "Sravan R",                      "level": "Engineer 1",        "allocs": {"CLX": 25, "CyRo": 25, "CyNoid": 50}},
    {"name": "Varikoti Siva Sai Dhanush",     "level": "Engineer 2",        "allocs": {}},
    {"name": "Yuvaraj Pandurang Badiger",     "level": "Engineer 3",        "allocs": {"Research/Exploration - HW": 25, "Neuroscience, Vision Research IISC": 10, "Hiring": 65}},
    {"name": "Timothee Hirt",                 "level": "Engineer 2",        "allocs": {"Audi(Door Assembly)": 50, "Robotics Research": 45, "Documentation": 5}},
    {"name": "Michael Bombile",               "level": "Principal Engineer", "allocs": {"CyRo": 10, "Audi(Door Assembly)": 10, "AMAT -1 (Kitting)": 15, "Robotics Research": 75}},
    {"name": "Hossein Afshari",               "level": "Principal Engineer", "allocs": {"CyNoid": 5, "SW Tools": 20, "Research/Exploration - SW": 20, "Code Refactoring": 5, "DevOps": 50}},
    {"name": "Sydney Hauke",                  "level": "Engineer 3",        "allocs": {"SW Tools": 50, "Research/Exploration - SW": 35, "Demo Development": 5, "DevOps": 5, "Hiring": 5}},
    {"name": "Shushuai Li",                   "level": "Senior Engineer",   "allocs": {"CLX": 30, "AMAT -1 (Kitting)": 20, "SW Tools": 20, "Robotics Research": 30}},
    {"name": "Advaith Sriram",                "level": "Engineer Intern",   "allocs": {"Robotics Research": 100}},
    {"name": "Oussama Jaffal",                "level": "Engineer Intern",   "allocs": {"Robotics Research": 100}},
]


# ── Project indices for dependency matrix ────────────────────────────────────
# 0:Audi  1:Lam  2:AMAT-1  3:AMAT-2  4:Schneider  5:Hyundai
# 6:CyRo  7:CyNoid  8:CLX  9:Robotics
# 10:SW Tools  11:HW Tools  12:Gripper
# 13:DevOps  14:Code Refactoring  15:Demo Dev
# 16:R&E HW  17:R&E SW  18:Neuroscience
# 19:Product Packaging  20:Documentation  21:Hiring

# PHI[j,k] — project k's completion gates project j's progress rate
PHI_EDGES = [
    # Solutions ← Products
    (0, 6, 0.8),   # Audi ← CyRo
    (1, 6, 0.8),   # Lam ← CyRo
    (2, 6, 0.8),   # AMAT-1 ← CyRo
    (3, 6, 0.7),   # AMAT-2 ← CyRo
    (4, 7, 0.7),   # Schneider ← CyNoid
    (5, 7, 0.7),   # Hyundai ← CyNoid
    # Solutions ← Demo Dev (customer iteration speed)
    (0, 15, 0.2), (1, 15, 0.2), (2, 15, 0.2),
    (3, 15, 0.2), (4, 15, 0.2), (5, 15, 0.2),
    # Products ← Platform
    (6,  8, 0.6),  # CyRo ← CLX
    (7,  8, 0.6),  # CyNoid ← CLX
    # Products ← Gripper (hardware component)
    (6, 12, 0.5),  # CyRo ← Gripper
    (7, 12, 0.4),  # CyNoid ← Gripper
    # Products ← Research
    (6,  9, 0.3),  # CyRo ← Robotics Research
    (7,  9, 0.3),  # CyNoid ← Robotics Research
    # Platform ← SW Tools
    (8, 10, 0.2),  # CLX ← SW Tools
    # Research ← Foundational R&E
    (9, 16, 0.4),  # Robotics ← R&E HW
    (9, 17, 0.3),  # Robotics ← R&E SW
    (9, 18, 0.2),  # Robotics ← Neuroscience/IISC
]

# ALPHA[j,k] — project k's completion boosts efficiency of people on project j
ALPHA_EDGES = [
    # Documentation boosts everyone (universal knowledge)
    *[(j, 20, 0.10) for j in range(22)],
    # SW Tools boosts all SW-heavy projects
    *[(j, 10, 0.30) for j in [0,1,2,3,4,5,6,7,8,13,14,15,17]],
    # HW Tools boosts hardware projects
    *[(j, 11, 0.30) for j in [9,11,12,16]],
    # DevOps boosts CI/CD-dependent projects
    *[(j, 13, 0.20) for j in [6,7,8,10,14,15,17]],
    # Code Refactoring boosts SW projects
    *[(j, 14, 0.20) for j in [6,7,8,10,15,17]],
    # Product Packaging boosts solutions deployment
    *[(j, 19, 0.15) for j in [0,1,2,3,4,5]],
    # Demo Dev boosts solutions customer iteration
    *[(j, 15, 0.15) for j in [0,1,2,3,4,5]],
]


def build_tensors():
    """
    Returns:
        P     [n,m]  allocation fractions
        CAP   [n]    capability scores
        W     [m]    resistance (difficulty) — higher = harder
        X     [n,m]  eligibility mask
        H     [n,m]  base efficiency (default 1.0)
        G     [m]    project goals
        PHI   [m,m]  progress coupling (gates)
        ALPHA [m,m]  efficiency coupling (boosts)
        names        list[str]
        proj_keys    list[str]
    """
    n = len(PEOPLE)
    m = len(PROJECTS)
    proj_index = {key: j for j, (key, _) in enumerate(PROJECTS)}

    P   = np.zeros((n, m), dtype=np.float32)
    X   = np.zeros((n, m), dtype=np.float32)
    CAP = np.zeros(n,      dtype=np.float32)
    W   = np.array([w for _, w in PROJECTS], dtype=np.float32)
    H   = np.ones((n, m),  dtype=np.float32)
    G   = np.ones(m,       dtype=np.float32)  # set in server after bootstrap

    PHI   = np.zeros((m, m), dtype=np.float32)
    ALPHA = np.zeros((m, m), dtype=np.float32)

    for j, k, v in PHI_EDGES:
        if j < m and k < m:
            PHI[j, k] = v
    for j, k, v in ALPHA_EDGES:
        if j < m and k < m:
            ALPHA[j, k] = max(ALPHA[j, k], v)  # take max if duplicate

    names = []
    for i, person in enumerate(PEOPLE):
        names.append(person["name"])
        CAP[i] = CAP_BY_LEVEL.get(person["level"], 60.0)
        total = sum(person["allocs"].values()) or 1.0
        for proj_key, pct in person["allocs"].items():
            j = proj_index.get(proj_key)
            if j is not None:
                P[i, j] = pct / total
                X[i, j] = 1.0

    proj_keys = [k for k, _ in PROJECTS]
    return P, CAP, W, X, H, G, PHI, ALPHA, names, proj_keys
