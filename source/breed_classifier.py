"""품종 분류 담당: 동물 크롭을 Oxford 37품종 모델에 전달합니다.
설명할 함수: classify, choose_breed. 보류 기준은 config.BREED_CONFIDENCE입니다.
"""
from config import BREED_CONFIDENCE

CAT_BREEDS = {"Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
              "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"}
EXPECTED_CLASSES = (
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle",
    "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau",
    "English Cocker Spaniel", "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese",
    "Japanese Chin", "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard", "Samoyed",
    "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier",
)


def choose_breed(probabilities, classes, species, threshold):
    """37개 전체 softmax 값을 사용하며, 종별 재정규화를 하지 않습니다."""
    order = sorted(range(len(classes)), key=lambda i: float(probabilities[i]), reverse=True)
    best = order[0]
    candidate, score = classes[best], float(probabilities[best])
    predicted_species = "cat" if candidate in CAT_BREEDS else "dog"
    reason = "species_mismatch" if predicted_species != species else ("low_confidence" if score < threshold else "")
    return {"breed": "unknown" if reason else candidate, "candidate_breed": candidate,
            "breed_confidence": score, "status": "review" if reason else "classified", "reason": reason,
            "top3": [{"breed": classes[i], "score": float(probabilities[i])} for i in order[:3]]}



class BreedClassifier:
    def __init__(self, weights_path, confidence=BREED_CONFIDENCE):
        import torch
        from torchvision import models, transforms
        # 임의의 Python 객체를 실행하지 않는 제한된 가중치 로딩을 사용합니다.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.classes = checkpoint["classes"]
        if tuple(self.classes) != EXPECTED_CLASSES or checkpoint["num_classes"] != 37:
            raise ValueError("Oxford 37품종 모델의 클래스 정보가 예상과 다릅니다.")
        self.classifier = models.resnet50(weights=None)
        self.classifier.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(2048, 37))
        self.classifier.load_state_dict(checkpoint["model_state"], strict=True)
        self.classifier.eval()
        # 제공자가 학습/추론 예제에 명시한 전처리와 동일합니다.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.confidence = confidence

    def classify(self, crop, species):
        import torch
        tensor = self.transform(crop).unsqueeze(0)
        with torch.inference_mode():
            probabilities = self.classifier(tensor).softmax(dim=1)[0].tolist()
        return choose_breed(probabilities, self.classes, species, self.confidence)
