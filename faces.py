import glob
import json
import os
import random
import tempfile
import torch
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image, ImageDraw, ExifTags, ImageFont
from retinaface import RetinaFace

from deepface import DeepFace
import cv2
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt

# from captioning import clip_embed


EXEMPLAR_PATH = "/mnt/f/Dreambooth/Faces/"


def open_image(filepath):
    image = Image.open(filepath)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = image._getexif()
        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError, TypeError):
        pass
    return image


def embed_face(image, box=None, extract_face=False, add_clip_embedding=False):
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    if box:
        image = image[box["top"]:box["top"]+box["height"], box["left"]:box["left"]+box["width"]]
    if extract_face:
        faceboxes = sorted(get_face_locations(image), key=lambda x: (x["top"], x["left"]))
        if len(faceboxes) == 0:
            return None
        facebox = faceboxes[0]
        image = image[facebox["top"]:facebox["top"]+facebox["height"], facebox["left"]:facebox["left"]+facebox["width"]]
    embedding = DeepFace.represent(image, model_name="VGG-Face", enforce_detection=False)
    if add_clip_embedding:
        embedding = embedding + clip_embed(image)
    return embedding


def get_exemplars():
    exemplars = {}
    for folder, _, filenames in list(os.walk(EXEMPLAR_PATH))[1:]:
        name = os.path.basename(folder)
        cache_path = os.path.join(folder, "cache.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                exemplars[name] = json.load(f)
                continue
        images = [f for f in filenames if f[-4:].lower() in [".jpg", ".png", ".jpeg"]]
        exemplars[name] = list(np.mean([embed_face(os.path.join(folder, f)) for f in images], axis=0))
        with open(cache_path, "w") as f:
            json.dump(exemplars[name], f)
    return exemplars


def get_exemplar_scores(embedding):
    scores = {}
    for name, exemplar in get_exemplars().items():
        scores[name] = cosine_similarity([embedding], [exemplar]).mean()
    return scores


def get_face_locations(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    face_detector = RetinaFace.build_model()
    results = RetinaFace.detect_faces(image, model=face_detector, threshold=0.9)
    if isinstance(results, tuple):
        return []
    return [_convert_box_format(f["facial_area"]) for f in results.values()]




def _convert_box_format(box, width=None, height=None):
    left1 = max(box[0], 0)
    top1 = max(box[1], 0)
    left2 = min(box[2], width) if width else box[2]
    top2 = min(box[3], height) if height else box[3]
    return {"left": round(left1), "top": round(top1), "width": round(left2-left1), "height": round(top2-top1)}


detr_feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

def detect_objects(image, category="person", threshold=0.9):

    if isinstance(category, str):
        category = detr_model.config.label2id[category]

    if isinstance(image, str):
        image = open_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    inputs = detr_feature_extractor(images=image, return_tensors="pt")
    outputs = detr_model(**inputs)

    # convert outputs (bounding boxes and class logits)
    target_sizes = torch.tensor([image.size[::-1]])
    results = detr_feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]

    include = (results["scores"] > threshold) & (results["labels"] == category)

    return [_convert_box_format(box, width=image.width, height=image.height) for box in results["boxes"][include].tolist()], results["scores"][include].tolist()


def calc_normalize_scores(scores):
    total = sum(scores.values())
    return {k: len(scores) * v/total for k, v in scores.items()}


def detect_and_label_people(image, category_threshold=0.8, category="person", exemplar_threshold=1, normalize_scores=True):
    boxes, scores = detect_objects(image, threshold=category_threshold, category=category)
    labels = []
    label_scores = []
    # if len(boxes) == 0:
    #     boxes = get_face_locations(image)
    #     if len(boxes) == 0:
    #         return [], [], []
    for box in boxes + get_face_locations(image):
        face_embedding = embed_face(image, box=box, extract_face=True)
        if face_embedding is None:
            continue
        scores = get_exemplar_scores(face_embedding)
        if normalize_scores:
            scores = calc_normalize_scores(scores)
        # del scores["a man"]
        # del scores["a woman"]
        if max(scores.values()) > exemplar_threshold:
            labels.append(max(scores, key=scores.get))
        else:
            labels.append("unknown")
        label_scores.append(scores)
    return boxes, labels, label_scores

def get_normalized_exemplar_scores(image, boxes, extract_face=False):
    scores = []
    for box in boxes:
        face_embedding = embed_face(image, box=box, extract_face=extract_face)
        if face_embedding is None:
            scores.append({})
        ex_scores = get_exemplar_scores(face_embedding)
        ex_scores = calc_normalize_scores(ex_scores)
        # del ex_scores["a man"]
        # del ex_scores["a woman"]
        scores.append(ex_scores)
    return scores


def get_most_likely_box_for_person(image, name):
    boxes, labels, label_scores = detect_and_label_people(image)
    match = None
    current_max = 0
    for box, scores in zip(boxes, label_scores):
        if scores[name] > current_max:
            match = box
            current_max = scores[name]
    return match


def crop_image_to_box(image, box):
    if isinstance(image, str):
        image = open_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.crop((box["left"], box["top"], box["left"]+box["width"], box["top"]+box["height"]))


def draw_boxes_on_image(image, boxes, labels=None, color=(0, 255, 0), thickness=2):
    if isinstance(image, str):
        image = open_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    else:
        image = image.copy()
    draw = ImageDraw.Draw(image)
    for box in boxes:
        if box is None:
            continue
        draw.rectangle([(box["left"], box["top"]), (box["left"]+box["width"], box["top"]+box["height"])], outline=color, width=thickness)
    if labels:
        for box, label in zip(boxes, labels):
            if box is None:
                continue
            draw.text((box["left"], box["top"]), label, fill=color)
    return image


def draw_text_on_image(image, text, box):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", image.width // 50)
    draw.text((box["left"], box["top"]), text, font=font)
    return image


def scale_box(box, factor, image=None):
    if image:
        factor = min(image.width / box["width"], image.height / box["height"], factor)
    factor = factor - 1
    scaled_box = {
        "left": box["left"] - box["width"] * factor / 2,
        "top": box["top"] - box["height"] * factor / 2,
        "width": box["width"] * (1 + factor),
        "height": box["height"] * (1 + factor),
    }
    if scaled_box["left"] < 0:
        scaled_box["left"] = 0
    if scaled_box["top"] < 0:
        scaled_box["top"] = 0
    if image:
        if scaled_box["left"] + scaled_box["width"] > image.width:
            scaled_box["width"] = image.width - scaled_box["left"]
        if scaled_box["top"] + scaled_box["height"] > image.height:
            scaled_box["height"] = image.height - scaled_box["top"]
    return scaled_box


def shift_box_to_center_of_inner_within_outer(box, innerbox, outerbox):
    # shift the box to the center of the innerbox
    box["left"] = innerbox["left"] + (innerbox["width"] - box["width"]) / 2
    box["top"] = innerbox["top"] + (innerbox["height"] - box["height"]) / 2
    # keep the box inside the original outerbox and in the image
    if box["left"] < outerbox["left"]:
        box["left"] = outerbox["left"]
    if box["top"] < outerbox["top"]:
        box["top"] = outerbox["top"]
    if box["left"] + box["width"] > outerbox["left"] + outerbox["width"]:
        box["left"] = outerbox["left"] + outerbox["width"] - box["width"]
    if box["top"] + box["height"] > outerbox["top"] + outerbox["height"]:
        box["top"] = outerbox["top"] + outerbox["height"] - box["height"]
    # if box["left"] < 0:
    #     box["left"] = 0
    # if box["top"] < 0:
    #     box["top"] = 0
    # if box["left"] + box["width"] > image.width:
    #     box["left"] = image.width - box["width"]
    # if box["top"] + box["height"] > image.height:
    #     box["top"] = image.height - box["height"]
    return box


def shrink_to_fit(box, innerbox, outerbox, avoidboxes=None, shift=True):
    for _ in range(100000):
        if shift:
            box = shift_box_to_center_of_inner_within_outer(box, innerbox, outerbox)
        overlaps_with_avoidboxes = any([box_portion_contained(box, avoidbox) > 0.03 for avoidbox in avoidboxes]) if avoidboxes else False
        wider_than_innerbox = box["width"] > innerbox["width"]
        taller_than_innerbox = box["height"] > innerbox["height"]
        wider_than_outerbox = box["width"] > outerbox["width"]
        taller_than_outerbox = box["height"] > outerbox["height"]
        if not wider_than_innerbox or not taller_than_innerbox:
            break  # box is smaller than innerbox, so we stop shrinking
        if (not overlaps_with_avoidboxes) and (not wider_than_outerbox) and (not taller_than_outerbox):
            break  # box is not overlapping with avoidboxes and is smaller than outerbox, so we stop shrinking
        box = scale_box(box, 0.99)
    return box


def find_optimal_crop_boxes(image, face, body, otherfaces, padding=1.1, small_face_threshold=0.2):
    
    if isinstance(image, str):
        image = open_image(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    imagebox = {"left": 0, "top": 0, "width": image.width, "height": image.height}

    # don't worry about avoiding faces that are too small
    otherfaces = [otherface for otherface in otherfaces if otherface["width"] > small_face_threshold * face["width"]]

    # remember the original boxes, pre-cropping
    original_body = body.copy()
    original_face = face.copy()

    # scale the boxes for target padding
    face = scale_box(face, padding, image)
    body = scale_box(body, padding, image)
    
    # convert body into squares and ensure they fit in the image
    body_cropped = contract_to_square(body)
    body_cropped = shrink_to_fit(box=body_cropped, innerbox=face, outerbox=imagebox)
    body_expanded = expand_to_square(body)
    body_expanded = shrink_to_fit(box=body_expanded, innerbox=face, outerbox=imagebox)

    # expand face into a square, and shrink it to fit in the body
    face = expand_to_square(face)
    face = shrink_to_fit(face, innerbox=scale_box(original_face, 0.5), outerbox=body)
    
    square_dim = min(image.width, image.height)
    fullsquarebox = {"left": 0, "top": 0, "width": square_dim, "height": square_dim}
    fullsquarebox_centered_on_face = shift_box_to_center_of_inner_within_outer(fullsquarebox, face, imagebox)
    fullsquarebox_centered_on_body = shift_box_to_center_of_inner_within_outer(fullsquarebox, body, imagebox)

    return {
        "face": face,
        "face_expanded_avoiding": shrink_to_fit(scale_box(face, 1.25), innerbox=original_face, outerbox=imagebox, avoidboxes=otherfaces),
        "face_doubled_avoiding": shrink_to_fit(scale_box(face, 2), innerbox=original_face, outerbox=imagebox, avoidboxes=otherfaces),
        "face_tripled_avoiding": shrink_to_fit(scale_box(face, 3), innerbox=original_face, outerbox=imagebox, avoidboxes=otherfaces),
        "body_cropped_on_face": body_cropped,
        "body_expanded_on_face": body_expanded,
        "body_cropped_on_face_avoiding": shrink_to_fit(body_cropped, innerbox=face, outerbox=imagebox, avoidboxes=otherfaces),
        "body_expanded_on_face_avoiding": shrink_to_fit(body_expanded, innerbox=face, outerbox=imagebox, avoidboxes=otherfaces),
        "full_centered_on_face": fullsquarebox_centered_on_face,
        "full_centered_on_face_avoiding": shrink_to_fit(fullsquarebox_centered_on_face, innerbox=face, outerbox=imagebox, avoidboxes=otherfaces),
        "full_centered_on_body": fullsquarebox_centered_on_body,
        "full_centered_on_body_avoiding": shrink_to_fit(fullsquarebox_centered_on_body, innerbox=body, outerbox=imagebox, avoidboxes=otherfaces),
    }


def expand_to_square(box):
    if box["width"] > box["height"]:
        box["top"] -= (box["width"] - box["height"]) / 2
        box["height"] = box["width"]
    else:
        box["left"] -= (box["height"] - box["width"]) / 2
        box["width"] = box["height"]
    return box


def contract_to_square(box):
    if box["width"] > box["height"]:
        box["left"] += (box["width"] - box["height"]) / 2
        box["width"] = box["height"]
    else:
        box["top"] += (box["height"] - box["width"]) / 2
        box["height"] = box["width"]
    return box


def area(box):
    return box["width"] * box["height"]


def box_portion_contained(contained, container):
    if contained is None or container is None:
        return 0
    intersection = get_intersection_box(contained, container)
    return intersection["width"] * intersection["height"] / (contained["width"] * contained["height"])
    

def get_intersection_box(box, other_box):
    left = max(box["left"], other_box["left"])
    top = max(box["top"], other_box["top"])
    right = min(box["left"] + box["width"], other_box["left"] + other_box["width"])
    bottom = min(box["top"] + box["height"], other_box["top"] + other_box["height"])
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def get_box_with_greatest_overlap(box, boxes):
    max_overlap = 0
    max_box = None
    for other_box in boxes:
        overlap = box_portion_contained(box, other_box)
        bigger_overlap = overlap > max_overlap
        equal_overlap_but_bigger = overlap == max_overlap and overlap > 0 and area(other_box) > area(max_box)
        if bigger_overlap or equal_overlap_but_bigger:
            max_overlap = overlap
            max_box = other_box
    return max_box


def intersection_over_union(box, other_box):
    intersection = get_intersection_box(box, other_box)
    return area(intersection) / (area(box) + area(other_box) - area(intersection))

def identical_boxes(box, other_box, threshold=0.75):
    if intersection_over_union(box, other_box) < threshold:
        return False
    if threshold < (area(box) / area(other_box)) < 1 / threshold:
        return True
    else:
        return False


def remove_duplicate_dict_boxes(boxes, threshold=0.75):
    unique_boxes = {}
    for key, box in sorted(boxes.items(), key=lambda x: area(x[1]), reverse=True):
        if not any(identical_boxes(box, other_box, threshold) for other_box in unique_boxes.values()):
            unique_boxes[key] = box
    return unique_boxes



# image = open_image("/mnt/f/Dreambooth/Entities/Jamie/Raw/20210818_195230.jpg")
# jamiebox = get_most_likely_box_for_person(image, "jamiealexandre man", )
# peopleboxes, _ = detect_objects(image, threshold=0.8, category="person")
# otherpeopleboxes = [box for box in peopleboxes if box != jamiebox]
# faceboxes = get_face_locations(image)
# jamieface = get_box_with_greatest_overlap(jamiebox, faceboxes)
# otherfaceboxes = [box for box in faceboxes if box != jamieface]
# jamiecrops = find_optimal_crop_boxes(image, jamieface, jamiebox, otherfaceboxes)
# img = draw_boxes_on_image(image, otherpeopleboxes, color="green")
# img = draw_boxes_on_image(img, faceboxes, color="yellow")
# img = draw_boxes_on_image(img, [jamiecrops["face"]], color="yellow", thickness=5)
# img = draw_boxes_on_image(img, [jamiecrops["body_cropped_on_face"]], color="brown", thickness=35)
# img = draw_boxes_on_image(img, [jamiecrops["body_expanded_on_face"]], color="green", thickness=31)
# img = draw_boxes_on_image(img, [jamiecrops["body_cropped_on_face_avoiding"]], color="purple", thickness=27)
# img = draw_boxes_on_image(img, [jamiecrops["body_expanded_on_face_avoiding"]], color="orange", thickness=23)
# img = draw_boxes_on_image(img, [jamiecrops["full_centered_on_face"]], color="red", thickness=19)
# img = draw_boxes_on_image(img, [jamiecrops["full_centered_on_face_avoiding"]], color="white", thickness=15)
# img = draw_boxes_on_image(img, [jamiecrops["full_centered_on_body"]], color="pink", thickness=11)
# img = draw_boxes_on_image(img, [jamiecrops["full_centered_on_body_avoiding"]], color="black", thickness=7)
# tmpfile = tempfile.NamedTemporaryFile(suffix=".jpg")
# # crop_image_to_box(image, box).save(tmpfile.name)
# img.save(tmpfile.name)
# print(tmpfile.name)


    





def find_person(image, name="jamiealexandre man", face_size_threshold=0.03):
    
    # find people
    peopleboxes, _ = detect_objects(image, threshold=0.75, category="person")
    # find faces
    faceboxes = get_face_locations(image)
    # filter out small faces
    max_face_size = max([area(fb) for fb in faceboxes])
    faceboxes = [fb for fb in faceboxes if area(fb) > max_face_size * face_size_threshold]

    # get the best matching person box for each face
    peopleboxes_for_faces = []
    for facebox in faceboxes:
        maxbox = facebox
        maxoverlap = 0
        for peoplebox in peopleboxes:
            overlap = box_portion_contained(facebox, peoplebox)
            if overlap > maxoverlap:
                maxoverlap = overlap
                maxbox = peoplebox
        peopleboxes_for_faces.append(maxbox)

    facescores = get_normalized_exemplar_scores(image, faceboxes, extract_face=False)

    maxscore = 0
    maxfacebox = None
    maxpersonbox = None
    for personbox, facebox, facescore in zip(peopleboxes_for_faces, faceboxes, facescores):
        if personbox is None:
            continue
        if facescore[name] > maxscore:
            maxscore = facescore[name]
            maxpersonbox = personbox
            maxfacebox = facebox
    
    return maxfacebox, maxpersonbox, peopleboxes_for_faces, faceboxes, facescores



def draw_faces_and_people(image, name, maxfacebox, maxpersonbox, peopleboxes_for_faces, faceboxes, facescores):
    img = draw_boxes_on_image(image, peopleboxes_for_faces, color="green")
    img = draw_boxes_on_image(img, faceboxes, color="yellow")
    img = draw_boxes_on_image(img, [maxfacebox], color="yellow", thickness=10)
    img = draw_boxes_on_image(img, [maxpersonbox], color="green", thickness=10)
    for facebox, facescore in zip(faceboxes, facescores):
        img = draw_text_on_image(img, f"{facescore[name]:.2f}", facebox)
    return img

def draw_crops(image, cropdict):
    image = draw_boxes_on_image(image, [cropdict.get("face")], color="yellow", thickness=43)
    image = draw_boxes_on_image(image, [cropdict.get("face_expanded_avoiding")], color="rgb(0,0,50)", thickness=39)
    image = draw_boxes_on_image(image, [cropdict.get("face_doubled_avoiding")], color="rgb(0,50,0)", thickness=39)
    image = draw_boxes_on_image(image, [cropdict.get("face_tripled_avoiding")], color="rgb(50,0,0)", thickness=39)
    image = draw_boxes_on_image(image, [cropdict.get("body_cropped_on_face")], color="black", thickness=35)
    image = draw_boxes_on_image(image, [cropdict.get("body_expanded_on_face")], color="green", thickness=31)
    image = draw_boxes_on_image(image, [cropdict.get("body_cropped_on_face_avoiding")], color="purple", thickness=27)
    image = draw_boxes_on_image(image, [cropdict.get("body_expanded_on_face_avoiding")], color="orange", thickness=23)
    image = draw_boxes_on_image(image, [cropdict.get("full_centered_on_face")], color="red", thickness=19)
    image = draw_boxes_on_image(image, [cropdict.get("full_centered_on_face_avoiding")], color="white", thickness=15)
    image = draw_boxes_on_image(image, [cropdict.get("full_centered_on_body")], color="pink", thickness=11)
    image = draw_boxes_on_image(image, [cropdict.get("full_centered_on_body_avoiding")], color="brown", thickness=7)
    return image


def find_faces_and_draw_crops(image, name):
    maxfacebox, maxpersonbox, peopleboxes_for_faces, faceboxes, facescores = find_person(image, name)
    crops = get_crops_for_face(image, name)
    # image = draw_faces_and_people(image, name, maxfacebox, maxpersonbox, peopleboxes_for_faces, faceboxes, facescores)
    image = draw_crops(image, crops)
    return image


def get_crops_for_face(image, name):
    maxfacebox, maxpersonbox, peopleboxes_for_faces, faceboxes, facescores = find_person(image, name)
    crops = find_optimal_crop_boxes(image, maxfacebox, maxpersonbox, [facebox for facebox in faceboxes if facebox != maxfacebox])
    if len(faceboxes) > 1:
        # remove all the crops that are full-image, if there are other faces
        for key in list(crops.keys()):
            if key.startswith("full") and not key.endswith("avoiding"):
                del crops[key]
    crops = remove_duplicate_dict_boxes(crops)
    return crops

# tmpfile = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
# image.save(tmpfile.name)
# print(tmpfile.name)

name = "jamiealexandre man"
for path in list(glob.glob("/mnt/f/Dreambooth/Entities/Jamie/Raw/*.jpg")):
    # if "IMG_20180728_192957.jpg" not in path:
    #     continue
    image = open_image(path)
    image = find_faces_and_draw_crops(image, name)
    tmpfile = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    image.save(tmpfile.name)
    print(tmpfile.name)
