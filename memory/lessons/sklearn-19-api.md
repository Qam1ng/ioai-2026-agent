# sklearn 1.9 removed LogisticRegression multi_class kwarg

LogisticRegression(multi_class=...) raises TypeError on sklearn>=1.9 (multinomial
is the default now). Also: liblinear is binary/ovr only. Prefer lbfgs +
class_weight='balanced' for imbalanced multiclass linear probes.
