RELATION_PATTERNS = [
    {
        "relation_type": "is_a",
        "verbs": ["be"],  # lemma for "is", "are", "was"
        "allowed": [("Concept", "Concept")],
    },
    {
        "relation_type": "improves",
        "verbs": ["improve", "increase", "boost", "outperform"],  # lemmas
        "allowed": [("Model", "Metric"), ("Method", "Metric")],
    },
    {
        "relation_type": "evaluated_on",
        "verbs": ["evaluate", "test"],  # lemmas for "evaluated", "tested"
        "allowed": [("Model", "Dataset"), ("Method", "Dataset")],
    },
    {
        "relation_type": "solves",
        "verbs": ["solve", "address"],  # lemmas
        "allowed": [("Model", "Task"), ("Method", "Task")],
    },
    {
        "relation_type": "depends_on",
        "verbs": ["depend", "require"],  # lemmas
        "allowed": "any",
    }
]
