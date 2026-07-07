# Knowledge Distillation Notes

Knowledge distillation trains a compact student to imitate a stronger teacher.
For classification, the student learns from two signals:

1. Hard labels: the normal cross-entropy target.
2. Soft labels: the teacher probability distribution after temperature scaling.

The soft target is useful because it contains class similarity information. For
example, an image of a truck may receive some probability mass for automobile.
That "dark knowledge" is missing from one-hot labels.

This project adds two vision-specific signals:

- Feature alignment: project student tokens into the teacher token dimension and
  match them with MSE.
- Relation alignment: compare token-token cosine relation matrices. This is
  more architecture tolerant than directly matching every channel.

The total loss is:

```text
L = ce_weight * CE(student, label)
  + kd_weight * T^2 * KL(student / T, teacher / T)
  + feature_weight * MSE(project(student_tokens), teacher_tokens)
  + relation_weight * MSE(token_relations_student, token_relations_teacher)
```

## Why Transformer to Mamba?

A ViT teacher mixes all patch tokens with attention. The Mamba-style student
uses a recurrent state-space scan over tokens, which has a different inductive
bias and better scaling potential for long sequences. Distillation helps the
student inherit global semantic behavior without copying the teacher's exact
attention mechanism.

## Where to Extend

- Replace `MambaMixer` with optimized `mamba-ssm` kernels.
- Add intermediate-layer distillation from selected teacher/student depths.
- For segmentation, replace the classification head with a decoder and distill
  logits/features at pixel or patch level.
- For cross-domain medical images, add an alignment module between teacher and
  student features, then distill relation graphs between original and augmented
  samples.
