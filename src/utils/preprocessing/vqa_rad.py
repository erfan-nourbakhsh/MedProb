import os
import os.path as osp
import re
import json

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

import warnings

warnings.filterwarnings("ignore")
VQA_RAD_DIR = "/raid/rsq813/MedAG/FinalProject/Data/VQA-RAD/osfstorage"
OUTPUT_DIR = "/raid/rsq813/MedAG/FinalProject/Data/VQA-RAD/Preprocessed"
YES_NO_ONLY = False
SEED: int = 42


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vqa_rad_df = pd.read_json(osp.join(VQA_RAD_DIR, "VQA_RAD Dataset Public.json"))
    vqa_rad_df["answer"] = vqa_rad_df["answer"].apply(lambda x: str(x).lower().strip())
    vqa_rad_df["answer_type"] = vqa_rad_df["answer_type"].apply(
        lambda x: str(x).lower().strip()
    )
    df = vqa_rad_df[vqa_rad_df["answer_type"] == "closed"]
    train_df = df[df["phrase_type"].isin(["freeform", "para"])]
    test_df = df[df["phrase_type"].isin(["test_freeform", "test_para"])]
    subset_cols = ["image_name", "question", "answer"]
    train_df = train_df.drop_duplicates(subset=subset_cols, ignore_index=True)
    test_df = test_df.drop_duplicates(subset=subset_cols, ignore_index=True)
    train_df = train_df[
        ~train_df[subset_cols]
        .apply(tuple, axis=1)
        .isin(test_df[subset_cols].apply(tuple, axis=1))
    ]
    _clean = lambda x: re.sub(" +", " ", str(x).lower()).replace(" ?", "?").strip()

    for split_df in [train_df, test_df]:
        split_df["question"] = split_df["question"].apply(_clean)
        split_df["answer"] = split_df["answer"].apply(_clean)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    QUESTION_OPTIONS: list[tuple[str, list[str]]] = [
        ("are there multiple or just 1 metastatic focus?", ["one", "multiple"]),
        (
            "is the colon more prominent on the patient's right or left side?",
            ["right", "left"],
        ),
        (
            "is the heart size in this image smaller or larger than if the image was taken ap?",
            ["smaller", "larger"],
        ),
        ("is this supratentorial or infratentorial?", ["supratentorial", "infratentorial"]),
        ("is this an mri or a ct scan?", ["mri", "ct"]),
        ("is the lesion on the left or right side of the brain?", ["left", "right"]),
        ("is the patient a female or male?", ["female", "male"]),
        ("is this patient male or female?", ["female", "male"]),
        ("is the lesion on the left or right?", ["left", "right"]),
        ("are the structures in the pancreas cystic or solid?", ["cystic", "solid"]),
        ("is this image normal or abnormal?", ["normal", "abnormal"]),
        (
            "is the gastric bubble shown on the left or right side of the patient?",
            ["left side", "right side"],
        ),
        (
            "are pleural opacities located on the left, right, or both sides of the lung?",
            ["left", "right", "both"],
        ),
        ("are the pleural opacities bilateral or unilateral?", ["bilateral", "unilateral"]),
        ("is this an axial or saggital view of the brain?", ["axial", "saggital"]),
        ("is this a t1 weighted, t2 weighted, or flair image?", ["t1", "t2", "flair"]),
        (
            "is there hypoinflation or hyperinflation of the lung?",
            ["hypoinflation", "hyperinflation"],
        ),
        (
            "is there hypoinflation or hyperinflation of the lung",
            ["hypoinflation", "hyperinflation"],
        ),
        ("are the small bubbles of air seen normal or abnormal?", ["normal", "abnormal"]),
        (
            "the small bubbles of air seen in the lumen are normal or abnormal?",
            ["normal", "abnormal"],
        ),
        ("is this a ct or an mri?", ["ct", "mri"]),
        ("was a ct or mri used to take the above image?", ["ct", "mri"]),
        (
            "is the stomach thickening regular and uniform or ragged/asymmetrical?",
            ["uniform", "asymmetric"],
        ),
        (
            "is the stomach wall thickening symmetric or asymmetric?",
            ["symmetric", "asymmetric"],
        ),
        ("is the abnormality focal or diffuse?", ["focal", "diffuse"]),
        ("is the abnormality hyper dense or hypo dense?", ["hyperdense", "hypodense"]),
        ("is the hila normal or enlarged?", ["normal", "enlarged"]),
        (
            "are the calcifications superior or inferior to the diaphragm?",
            ["superior", "inferior"],
        ),
        (
            "are the bowel loops on the right or left side of the patient?",
            ["right side", "left side"],
        ),
        (
            "is the pathologic part of this image the small bowel or colon?",
            ["small bowel", "colon"],
        ),
        ("is the lesion on the patient's right or left side?", ["right side", "left side"]),
        (
            "is the largest cyst in the left or right kidney?",
            ["left kidney", "right kidney"],
        ),
        ("was oral or iv contrast used?", ["oral", "iv", "both"]),
        (
            "are the lesions in the image more or less dense than surrounding tissue?",
            ["more dense", "less dense"],
        ),
        (
            "is the primary abnormality more or less dense than surrounding matter?",
            ["more dense", "less dense"],
        ),
        (
            "what is denser, the mass or the surrounding brain tissue?",
            ["mass", "surrounding tissue"],
        ),
        (
            "is the lesion in this image more or less dense than the surrounding tissue?",
            ["more dense", "less dense"],
        ),
        (
            "what is denser, the lesion or the surrounding tissue?",
            ["lesion", "the surrounding tissue"],
        ),
        (
            "is the largest air collection on the patient's left or the right side?",
            ["left side", "right side"],
        ),
        ("are there multiple masses or just a single big one?", ["just one", "multiple"]),
        ("is this a solid or cystic lesion?", ["solid", "cystic"]),
        ("is the lesion a solid or cystic lesion?", ["solid", "cystic"]),
        ("are the dark areas grey or white matter?", ["grey matter", "white matter"]),
        ("is this an abscess or cancer?", ["abscess", "cancer"]),
        (
            "do you suspect a physical injury or a medical process?",
            ["physical injury", "medical process"],
        ),
        (
            "what is the cause of this finding: medical process or physical injury?",
            ["medical process", "physical injury"],
        ),
        ("was this image taken in the ap or pa plane?", ["ap", "pa"]),
        (
            "was this a contrast ct or a non-contrast ct?",
            ["contrast ct with gi and iv contrast", "non-contrast ct"],
        ),
        ("is this a contrast ct or a non-contrast ct?", ["contrast ct", "non-contrast ct"]),
        ("is the lesion located in gray or white matter?", ["gray matter", "white matter"]),
        (
            "is the lesion seen in the gray or white matter?",
            ["gray matter", "white matter"],
        ),
        (
            "was this image taken with or without contrast?",
            ["with contrast", "without contrast"],
        ),
        (
            "is the consistency of the abscess located in the left upper quadrant homogeneous or heterogeneous?",
            ["homogeneous", "heterogeneous"],
        ),
        (
            "is the abscess in the left upper quadrant homogenous or heterogenous?",
            ["homogenous", "heterogeneous"],
        ),
        (
            "is this mri with contrast or without contrast?",
            ["with contrast", "without contrast"],
        ),
        (
            "was this mri taken with or without contrast?",
            ["with contrast", "without contrast"],
        ),
        (
            "are the pulmonary nodules diffuse in the chest or lateralized to one side?",
            ["diffuse", "lateralized"],
        ),
        (
            "is the location of the pulmonary nodules diffuse or lateralized to one side?",
            ["diffuse", "lateralized"],
        ),
        (
            "in which lobe are the lesions?",
            ["left frontal lobe", "right frontal lobe", "bilateral frontal lobes"],
        ),
        ("is the mass hyperintense or hypointense?", ["hyperintense", "hypointense"]),
        ("is this a pa or ap film?", ["pa", "ap"]),
        ("does this image use contrast or not?", ["contrast", "no contrast"]),
        ("does this ct have contrast or no contrast?", ["contrast", "no contrast"]),
        ("is this a singular or multilobulated lesion?", ["singular", "multilobulated"]),
        ("is the mass in the left or right side?", ["left", "right"]),
        (
            "is the contrast in the bowels or the vasculature?",
            ["in the bowels", "in the vasculature"],
        ),
        (
            "is the mass heterogeneous or homogeneous in appearance?",
            ["heterogeneous", "homogeneous"],
        ),
        ("is the csf radiolucent or radioopaque?", ["radiolucent", "radioopaque"]),
        ("was this image taken with an mri or ct scanner?", ["mri", "ct"]),
        ("is the csf enhanced or non enhanced?", ["enhanced", "non-enhanced"]),
        ("is this an ap or pa film?", ["ap", "pa"]),
        ("does the appendix appear normal or abnormal?", ["normal", "abnormal"]),
        ("is there appendix normal or abnormal in appearance?", ["normal", "abnormal"]),
        (
            "are the ground glass opacities located more in the apex or base of the lung?",
            ["apex", "base"],
        ),
        ("is this a cystic or solid mass?", ["cystic", "solid"]),
        ("is this a contrast or non contrast ct?", ["contrast", "non-contrast"]),
        ("is this a pneumonia vs. pleural effusion?", ["pneumonia", "pleural effusion"]),
        (
            "is a pneumonia or pleural effusion seen in this image?",
            ["pneumonia", "pleural effusion"],
        ),
        (
            "is the pathology seen hyperintense or hypointense in nature?",
            ["hyperintense", "hypointense"],
        ),
        (
            "is the abnormality hyperintense or hypointense?",
            ["hyperintense", "hypointense"],
        ),
        ("is this image a ct or mri image?", ["ct", "mri"]),
        (
            "is free air present under the patient's left or right hemidiaphragm?",
            ["left", "right"],
        ),
        ("is the trachea deviated to the right or left?", ["right", "left"]),
        ("do you suspect vascular process or a genetic process?", ["vascular", "genetic"]),
        ("is the etiology genetic or vascular?", ["genetic", "vascular"]),
        (
            "would you expect pleural plaques on other pleural surfaces vs just the hemithoraces?",
            ["yes", "no", "not sure"],
        ),
        (
            "do you expect the patient to have plaques on other pleura as well?",
            ["yes", "no", "maybe"],
        ),
        (
            "is gray or white matter highlighted in this image?",
            ["gray matter", "white matter"],
        ),
        (
            "which is highlighted in this image, white or gray matter?",
            ["white matter", "gray matter"],
        ),
        ("is this an infectious process?", ["yes", "no", "maybe"]),
        ("is the hyperinflation unilateral or bilateral?", ["unilateral", "bilateral"]),
        ("is the lung hyperinflated on one or both sides?", ["one side", "both sides"]),
        (
            "are the lines seen on the exterior or interior of the patient?",
            ["exterior", "interior"],
        ),
        (
            "are the lines in the image inside or outside of the patient?",
            ["inside", "outside"],
        ),
        (
            "is the mass in the liver regular or irregular in contour?",
            ["regular", "irregular"],
        ),
        (
            "is the circular opacity (located in the middle of this image) found on top of the patient or within the patient?",
            ["on top of the patient", "within the patient"],
        ),
        (
            "is the opacity located in the middle of the image inside the patient or superficial to the patient's skin?",
            ["inside the patient", "superficial to the patient's skin"],
        ),
        ("is the apical aeration normal or decreased?", ["normal", "decreased"]),
        ("is this image modality t1, t2, or flair?", ["t1", "t2", "flair"]),
        ("is the lesion on the right or left side of the brain?", ["right", "left"]),
    ]
    _OPTIONS_LOOKUP: dict[str, list[str]] = {q: opts for q, opts in QUESTION_OPTIONS}

    def resolve_options(question: str, answer: str) -> list[str] | None:
        ans = answer.lower().strip()
        if ans in ("yes", "no"):
            return ["yes", "no"]
        if YES_NO_ONLY:
            return None
        q_lower = question.lower().strip()
        if q_lower in _OPTIONS_LOOKUP:
            return _OPTIONS_LOOKUP[q_lower]
        return None

    def convert(df: pd.DataFrame, split_name: str) -> list[dict]:
        records = []
        skipped_no_options = 0
        skipped_mismatch = 0
        for _, row in df.iterrows():
            question = row["question"].capitalize().strip()
            if not question.endswith("?"):
                question += "?"
            answer = row["answer"].lower().strip()
            options = resolve_options(question, answer)
            if options is None:
                skipped_no_options += 1
                print(f"[{split_name}] SKIP – no options resolved")
                print(f"  Q: {question}")
                print(f"  A: {answer}\n")
                continue
            if answer not in options:
                skipped_mismatch += 1
                print(f"[{split_name}] SKIP – answer not in options")
                print(f"  Q: {question}")
                print(f"  Options: {options}")
                print(f"  A: {answer}\n")
                continue
            records.append(
                dict(
                    id=osp.splitext(row["image_name"])[0],
                    image=row["image_name"],
                    question=question,
                    answer=answer,
                    options=options,
                )
            )
        if skipped_no_options:
            print(f"[{split_name}] Total skipped (no options): {skipped_no_options}")
        if skipped_mismatch:
            print(f"[{split_name}] Total skipped (answer ∉ options): {skipped_mismatch}")
        return records
    train_records = convert(train_df, "train")
    test_records = convert(test_df, "test")

    def write_jsonl(data: list[dict], path: str) -> None:
        with open(path, "w") as fh:
            for record in data:
                json.dump(record, fh)
                fh.write("\n")

    def read_jsonl(path: str) -> list[dict]:
        with open(path, "r") as fh:
            return [json.loads(line) for line in fh]
    train_idx, val_idx = train_test_split(
        np.arange(len(train_records)),
        test_size=0.2,
        random_state=SEED,
    )
    final_train_records = [train_records[i] for i in train_idx]
    val_records = [train_records[i] for i in val_idx]

    write_jsonl(final_train_records, osp.join(OUTPUT_DIR, "train.jsonl"))

    write_jsonl(val_records, osp.join(OUTPUT_DIR, "val.jsonl"))

    write_jsonl(test_records, osp.join(OUTPUT_DIR, "test.jsonl"))

    print("=" * 55)

    print("VQA-RAD  |  Closed-ended splits")

    print("=" * 55)

    print(f"  Train : {len(train_records):>5} samples")

    print(f"  Val   : {len(val_records):>5} samples")

    print(f"  Test  : {len(test_records):>5} samples")

    print(f"  Total : {len(train_records) + len(test_records):>5} samples")

    print("=" * 55)

    with open(osp.join(OUTPUT_DIR, "dataset_statistics.txt"), "w") as f:
        f.write(f"Train: {len(train_records)}\n")
        f.write(f"Val: {len(val_records)}\n")
        f.write(f"Test: {len(test_records)}\n")
        f.write(f"Total: {len(train_records) + len(val_records) + len(test_records)}\n")
    import shutil

    VQA_RAD_IMAGE_SRC = (
        "/raid/rsq813/MedAG/FinalProject/Data/VQA-RAD/osfstorage/VQA_RAD Image Folder"
    )
    VQA_RAD_IMAGE_DST = osp.join(OUTPUT_DIR, "images")

    shutil.copytree(VQA_RAD_IMAGE_SRC, VQA_RAD_IMAGE_DST)

    print(f"  Copied images to : {VQA_RAD_IMAGE_DST}")

if __name__ == "__main__":
    main()
