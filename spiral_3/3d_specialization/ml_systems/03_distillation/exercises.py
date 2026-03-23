"""Knowledge Distillation Exercises."""

import torch

def distillation_loss(student_logits, teacher_logits, labels, temperature: float = 4.0, alpha: float = 0.5):
    """KD loss: alpha * KL(soft_teacher, soft_student) * T^2 + (1-alpha) * CE."""
    raise NotImplementedError

def soft_labels(teacher_logits, temperature: float):
    """Softened probability distribution."""
    raise NotImplementedError

def feature_distillation_loss(student_features, teacher_features):
    """MSE between intermediate representations."""
    raise NotImplementedError

def attention_transfer_loss(student_attention, teacher_attention):
    """Match attention maps."""
    raise NotImplementedError
