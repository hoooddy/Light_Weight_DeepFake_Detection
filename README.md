# 🏆 주요 성과

<aside>
💡

파라미터 수 **99.5%감소**, 추론 시간 **93.85% 개선**

</aside>

---

# 👨‍💻 개요 및 문제점 인식

딥페이크로 인한 **정치적, 사회적 혼란 발생**

- 딥페이크를 악용한 가짜뉴스는 유명 인사나 정치인을 포함한 사람들의 발언이나 행동을 왜곡하여 정치적, 사회적 혼란을 일으키고 있음

그 **진위 여부를 정확히 판별**하는 과정이 핵심

- 허위 정보가 확산되기 전에 **신속하게 진위를 판단**하고 차단하는 것이 중요

---

# 💽 데이터셋

META의  [`Deepfake Detection Challenge`](https://www.kaggle.com/c/body-morphometry-kidney-and-tumor) 활용

---

# 🖥️ 개발 및 개선 내용

### 1. 딥페이크 탐지 SOTA, Cross Efficient ViT 구현

- [Cross Efficient ViT](https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection)
- [Combinint EfficientNet and Vision Transformers for Video Deepfake Detection](https://arxiv.org/abs/2107.02612)

### 2. Knowledge Distillation 기반 모델 경량화

```python
class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, labels):
        # CrossEntropy Loss (Student)
        hard_loss = self.criterion(student_outputs, labels)
        
        # Soft Loss (Distillation)
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            torch.log_softmax(student_outputs / self.temperature, dim=0),
            torch.softmax(teacher_outputs / self.temperature, dim=0)
        )

        # Total Loss: Hard Loss + Soft Loss (with temperature scaling)
        total_loss = (1.0 - self.alpha) * hard_loss + self.alpha * soft_loss * (self.temperature ** 2)
        return total_loss
