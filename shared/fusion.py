"""Reconcile YOLO species with 37-breed softmax, without inventing probabilities."""
def choose_supported_breed(probabilities, classes, yolo_species, threshold, cat_breeds,
                           override_threshold=0.85, mass_threshold=0.90):
    if yolo_species not in ("cat", "dog"):
        raise ValueError("YOLO species must be cat or dog")
    if len(probabilities) != len(classes) or not classes:
        raise ValueError("Invalid breed score vector")
    scores = [float(x) for x in probabilities]
    if any(x < 0 or x > 1 for x in scores):
        raise ValueError("Breed probabilities must be in [0,1]")
    idx = sorted(range(len(classes)), key=lambda i: scores[i], reverse=True)
    best = idx[0]
    top_species = "cat" if classes[best] in cat_breeds else "dog"
    cat_mass = sum(scores[i] for i,name in enumerate(classes) if name in cat_breeds)
    dog_mass = sum(scores) - cat_mass
    top_mass = cat_mass if top_species == "cat" else dog_mass
    species = yolo_species
    if (top_species != yolo_species and scores[best] >= override_threshold
            and top_mass >= mass_threshold):
        species = top_species
        chosen = best
        reason = "species_corrected_high_confidence"
    else:
        allowed = [i for i in idx
                   if ("cat" if classes[i] in cat_breeds else "dog") == yolo_species]
        if not allowed:
            raise ValueError("Breed list has no candidates for YOLO species")
        chosen = allowed[0]
        reason = ("species_conflict" if top_species != yolo_species
                  else "low_confidence" if scores[chosen] < threshold else "")
    status = "classified" if not reason else "review"
    global3 = idx[:3]
    same3 = [i for i in idx if ("cat" if classes[i] in cat_breeds else "dog") == species][:3]
    format_top3 = lambda indices: " | ".join(f"{classes[i]} {scores[i]:.1%}" for i in indices)
    return {
        "top3_global": format_top3(global3),
        "top3_species": format_top3(same3),
        "species": species,
        "breed": classes[chosen] if status == "classified" else "unknown",
        "candidate_breed": classes[chosen],
        "breed_confidence": scores[chosen],
        "status": status,
        "reason": reason,
        "yolo_species": yolo_species,
        "resnet_species": top_species,
        "resnet_species_mass": top_mass,
        "yolo_species_mass": cat_mass if yolo_species == "cat" else dog_mass,
        "resnet_top_breed": classes[best],
        "resnet_top_confidence": scores[best],
        "top3": [{"breed":classes[i],"score":scores[i]} for i in idx[:3]],
        "top3_species_details": [
            {"breed":classes[i],"score":scores[i]}
            for i in idx if ("cat" if classes[i] in cat_breeds else "dog") == species
        ][:3]
    }
