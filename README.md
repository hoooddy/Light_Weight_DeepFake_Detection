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

<details>
  <summary>📝 클릭하여 내용 보기</summary>
> *모델이 큰 신경망으로 구성되어 있다면 배포에 많은 컴퓨팅 비용이 들 수 있다.* 
    > 
    당연한 말이지만 성능은 좋지만 모델이 크고, 복잡하다면 이를 활용하기엔 리소스가 제한적일 수 있다.
    
    - 복잡한 모델 Teacher: 예측 정확도 99% + 추론 시간 3시간
    - 단순한 모델 Student: 예측 정확도 90% + 추론 시간 3분
    
    위의 두 모델이 있을 때 어떤 모델이 서비스 활용에 적합할까? 물론 어떤 서비스인지에 따라 다를 수 있지만 대부분의 서비스 운영자들은 사용자의 가용성을 해치지 않기 위해 후자를 택할 것이다.
    
    약간의 비약이 있을 수 있지만 T모델의 정확도와 S 모델의 추론 시간을 가진다면 더 좋지 않을까?
    
    기존 모델과 성능은 유사하지만 그 크기를 좀 더 줄일 수 있는 여러 경량화 기법(Pruning, Quantization, …)들이 있지만 그 중 지식 증류(Knowledge Distillation)에 대해 알아보자.
    
    우선 이 `Knowledge`를 어떻게 정의할 수 있을까?
    
    <aside>
    💡
    
    **Knowledge란?**
    
    - 학습 과정에서 데이터를 통해 배운 **무언가**
    - Input과 Output의 **Mapping**
    </aside>
    
    경량화는 이 `Knowledge`를, 즉, 학습 과정에서 데이터를 통해 배운 **무언가**를 **보존**하면서, size를 줄이는 작업이다. 경량화 기법을 어떻게 디자인할 것인지는 `knowledge`를 어떻게 정의할지에 따라 다를 수 있다.
    
    예를 들어 **Parameter**를 보존하고 싶은 `Knowledge`로 정할 수 있다. Pruning과 Quantization은 이 Parameter를 최대한 보존하면서 경량화를 수행한다.
    
    <aside>
    💡
    
    **Pruning**
    
    가중치가 0에 가까운(영향력이 적은) 가중치를 제거하는 것이 목표
    
    **Quantization**
    
    실수(float) Parameter를 모방하는 정수(int) 값을 찾는 것이 목표
    
    </aside>
    
    위 두 경량화 기법들은 Parameter에 집중하기에 기존 구조에 제한 받는다.
    
    그렇기에 관점을 바꿔 **Output Vector**에 집중 해보자.
    
    - Output Vector에만 집중하기에 모델 구조에 제한적이지 않다.
        - 즉, 다른 구조를 활용해 Output Vector를 흉내낸다.
        - 예) 앙상블 같이 복잡한 모델의 Output Vector에 집중하여 작은 모델이 복잡한 모델의 Output Vector를 흉내낸다.
    
    이를 위해 시도해볼 수 있는 것이 지식 증류(Knowledge Distillation)이다.
    
    이제 증류(`Distillation`)을 어떻게 정의할 수 있을까?
    
    **혼합물로부터 순수한 물을 얻기 위한 과정**
    
    ![https://www.youtube.com/watch?v=NtZsrxrf_0k&t=1863s](https://prod-files-secure.s3.us-west-2.amazonaws.com/914754d5-747d-4110-a869-195f041398e0/9328a859-bca9-476e-89bc-f68d2ebad700/image.png)
    
    https://www.youtube.com/watch?v=NtZsrxrf_0k&t=1863s
    
    1. Bottle #1의 혼합물(Hard)을 가열(물리적 가열)한다.
    2. 증기가 되어(Soft) 관을 타고(Transfer) 넘어가면서 식는다(Cooling Off)
    3. Bottle #2에 순수한 물(Hard)이 담기게 된다.
        - 이때 Bottle #1과 Bottle #2의 모양(구조)이 달라도 전혀 상관없다.
    
    **복잡한 모델로부터 지식을 얻기 위한 과정**
    
    ![https://www.youtube.com/watch?v=NtZsrxrf_0k&t=1863s](https://prod-files-secure.s3.us-west-2.amazonaws.com/914754d5-747d-4110-a869-195f041398e0/7076694d-5f29-48f8-b82b-3da79c0295a4/image.png)
    
    https://www.youtube.com/watch?v=NtZsrxrf_0k&t=1863s
    
    1. Heavy DNN의 지식(Vector Output)을 가열(논리적 가열)한다.
        - 여기서 가열은 부드럽게 만들어준다 → 정규화 해준다 정도로만 이해해보자
    2. Soft 해진? 지식(Pure Knowledge)을 Light DNN으로 넘긴다.
    3. Light DNN에 Pure Knowledge가 담긴다.
    
    앞으로 쓰일 용어를 간단히 정리해보자.
    
    **1. Soft Label**
    
    일반적으로 분류 Task는 신경망의 마지막 레이어 softmax 레이어를 통해 각 클래스의 확률 값을 구한다.
    
    이때 **Teacher 모델**을 통해 얻을 수 있는 **확률 값**을 `Soft Label`이라 한다.
    
    예) [0.7, 0.2, 0.1]
    
    **2. Hard Label**
    
    정답 값으로 활용하는 값들을 `Hard Label` 이라 한다.
    
    예) [1, 0, 0]
    
    **3. Soft Loss**
    
    Teacher 모델의 출력과 Student 모델의 출력 간 차이, 이때 각 모델들의 출력은 softmax를 통해 도출된 확률 값들이다.
    
    **4. Hard Loss**
    
    Student 모델의 예측과 실제 레이블 간의 차이를 측정한 loss 값
    
    ![https://www.youtube.com/watch?v=NtZsrxrf_0k&t=1863s](https://prod-files-secure.s3.us-west-2.amazonaws.com/914754d5-747d-4110-a869-195f041398e0/2fe92d29-e8a3-4fbe-b0ec-a35c3cca3c86/image.png)
    
    https://www.youtube.com/watch?v=NtZsrxrf_0k&t=1863s
    
    위 그림의 Large DNN의 Output에 Softmax를 취해 확률 값을 얻을 수 있고, 확률 값이 가장 큰 Index가 정답이 될 것이다. 이때 **Softmax를 통해 얻은 확률 값**은 Cat이 Fish보다 높을 것이다. 이러한 정보 역시 `Knowledge`로 볼 수 있다.
    
    - 이러한 정보는 특정 이미지를 분류하는 데 도움이 되지 않지만, **이미지 구조를 비교하는 일반적인 능력**은 다른 테스트 이미지에 사용할 수 있습니다.
    
    ![https://www.youtube.com/watch?v=NtZsrxrf_0k&t=1863s](https://prod-files-secure.s3.us-west-2.amazonaws.com/914754d5-747d-4110-a869-195f041398e0/6a5b5369-f40c-4fee-8367-a2f05849a69f/image.png)
    
    https://www.youtube.com/watch?v=NtZsrxrf_0k&t=1863s
    
    하지만 여전히 큰 모델의 Soft Label은 Hard하다.
    
    - 학습이 잘 된 모델일 수록 정답 확률을 크게 계산 할 것이고, 아닌 것을 낮게 계산할 것이다.
    - 그렇기에 가열을 통해 부드럽게 만들어준다.
        - 부드럽게 만들어 준다 → **Soft Label을 Temperature라는 값으로 나누어 준다.**
        - 이때 Soft Label에 적용해준 Temperature 만큼 Student Model의 확률 값에도 나누어 준다.
    
    ![https://www.youtube.com/watch?v=NtZsrxrf_0k&t=1863s](https://prod-files-secure.s3.us-west-2.amazonaws.com/914754d5-747d-4110-a869-195f041398e0/b4bfcf9e-3fae-4cac-853d-0fbb99863445/image.png)
    
    https://www.youtube.com/watch?v=NtZsrxrf_0k&t=1863s
    
    **Temperature Setting**
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/914754d5-747d-4110-a869-195f041398e0/a590849c-b22a-4106-81e6-f20dfaf1a120/image.png)
    
    **큰 모델(Teacher Network)**로부터 **증류한 지식**을 **작은 모델(Student Network)**로 **Transfer**하는 일련의 과정
    
    ![https://intellabs.github.io/distiller/knowledge_distillation.html](https://prod-files-secure.s3.us-west-2.amazonaws.com/914754d5-747d-4110-a869-195f041398e0/31b606cd-8aaf-471e-a950-e82e82f6334f/image.png)
    
    https://intellabs.github.io/distiller/knowledge_distillation.html
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/914754d5-747d-4110-a869-195f041398e0/44281dcf-4a1f-4bd3-972c-108f80eec0de/image.png)
    
    복잡한 모델(T)와 단순한 모델(S)는 위 그림과 같은 특징을 지니고, Teacher, Student로 표현한다.
    
    **Teacher Network**
    
    - 앙상블 / 크고 일반적인 모델
    - 장점: 높은 성능
    - 단점: 비싼 컴퓨팅 비용
    - 제한적인 환경에서 배포 불가능
    
    **Student Network**
    
    - 작은 모델
    - 배포에 알맞음
    - 장점: 빠른 추론
    - 단점: T 보다 낮은 성능
    
    ![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/914754d5-747d-4110-a869-195f041398e0/ede19409-1075-4500-ba04-561c7e310536/image.png)
    
    따라서 최종 손실 함수는 아래와 같다.
    
    $Loss=\alpha*\text{Hard Loss} +(1-\alpha)*\text{Soft Loss}$
    
    - $\alpha$ → 두 손실 간의 가중치를 조정하는 하이퍼파라미터
    
    ### Temperature Scaling
    
    Teacher 모델의 Softmax 출력에 적용되는 하이퍼파라미터로 확률 분포를 부드럽게 만들어준다.
  

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
