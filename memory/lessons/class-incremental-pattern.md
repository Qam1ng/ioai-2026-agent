# Class-incremental tasks: reuse frozen-head signal, don't freeze mistakes

Winning pattern on the AST audio task: extract frozen-encoder features once and
cache; append the FROZEN old head's logits to the features; retrain a
class-balanced linear classifier over all classes. This preserves old-class
knowledge (logits as features) while letting balanced retraining fix classes the
old head got systematically wrong. Fully freezing the old head locks in its
errors; full unconstrained retraining forgets. ~10min budgets favor cached
features + linear/shallow heads over encoder fine-tuning.
