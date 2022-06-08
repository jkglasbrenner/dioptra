# This Software (Dioptra) is being made available as a public service by the
# National Institute of Standards and Technology (NIST), an Agency of the United
# States Department of Commerce. This software was developed in part by employees of
# NIST and in part by NIST contractors. Copyright in portions of this software that
# were developed by NIST contractors has been licensed or assigned to NIST. Pursuant
# to Title 17 United States Code Section 105, works of NIST employees are not
# subject to copyright protection in the United States. However, NIST may hold
# international copyright in software created by its employees and domestic
# copyright (or licensing rights) in portions of software that were assigned or
# licensed to NIST. To the extent that NIST holds copyright in this software, it is
# being made available under the Creative Commons Attribution 4.0 International
# license (CC BY 4.0). The disclaimers of the CC BY 4.0 license apply to all parts
# of the software developed or licensed by NIST.
#
# ACCESS THE FULL CC BY 4.0 LICENSE HERE:
# https://creativecommons.org/licenses/by/4.0/legalcode
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, TypeVar

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
import structlog
from imgaug.augmentables import BoundingBox, BoundingBoxesOnImage
from structlog.stdlib import BoundLogger

from dioptra import pyplugs
from dioptra.sdk.object_detection.architectures import YOLOV1ObjectDetector
from dioptra.sdk.object_detection.bounding_boxes.postprocessing import (
    BoundingBoxesYOLOV1PostProcessing,
)
from dioptra.sdk.object_detection.data import TensorflowObjectDetectionData

LOGGER: BoundLogger = structlog.stdlib.get_logger()

try:
    from tensorflow.data import Dataset

except ImportError:  # pragma: nocover
    LOGGER.warn(
        "Unable to import one or more optional packages, functionality may be reduced",
        package="tensorflow",
    )

try:
    from typing import TypedDict

except ImportError:
    from typing_extensions import TypedDict


T_coord = TypeVar("T_coord", bound=npt.NBitBase)
T_id = TypeVar("T_id", bound=npt.NBitBase)
T_score = TypeVar("T_score", bound=npt.NBitBase)


class COCOInfo(TypedDict):
    description: str
    version: str
    year: int


class COCOImage(TypedDict):
    id: int
    width: float
    height: float
    file_name: str
    date_captured: str


class COCOAnnotation(TypedDict):
    id: int
    image_id: int
    category_id: int
    bbox: list[float]
    iscrowd: int
    area: float


class COCOCategory(TypedDict):
    id: int
    name: str


class COCODataset(TypedDict):
    info: COCOInfo
    images: list[COCOImage]
    annotations: list[COCOAnnotation]
    catefories: list[COCOCategory]


class COCOResult(TypedDict):
    image_id: int
    category_id: int
    bbox: list[float]
    score: float


@pyplugs.register
def predict_bounding_boxes(
    estimator: YOLOV1ObjectDetector,
    dataset: TensorflowObjectDetectionData,
    bbox_postprocessing: BoundingBoxesYOLOV1PostProcessing,
    dataset_subset: str,
    coco_json_filepath: str | Path,
    batch_size: int,
    output_filepath: str | Path,
) -> str:
    output_filepath = Path(output_filepath)
    coco_results: Iterator[COCOResult] = _predict_bounding_boxes(
        model=estimator,
        dataset=dataset,
        bbox_postprocessing=bbox_postprocessing,
        dataset_subset=dataset_subset,
        coco_json_filepath=coco_json_filepath,
        batch_size=batch_size,
    )
    sorted_coco_results: list[COCOResult] = sorted(
        list(coco_results),
        key=lambda x: (x["image_id"], x["score"]),
    )

    with output_filepath.open("wt") as f:
        json.dump(obj=sorted_coco_results, fp=f)

    return str(output_filepath)


@pyplugs.register
def draw_bounding_box_on_images(
    images_dir: str | Path,
    coco_json_filepath: str | Path,
    coco_results_filepath: str | Path,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    images_with_bboxes: Iterator[dict[str, npt.NDArray]] = _draw_bounding_box_on_images(
        images_dir=images_dir,
        coco_json_filepath=coco_json_filepath,
        coco_results_filepath=coco_results_filepath,
    )

    for file_name, image_with_bboxes in images_with_bboxes.items():
        output_filepath = output_dir / file_name
        iio.imwrite(uri=output_filepath, image=image_with_bboxes)


def _predict_bounding_boxes(
    model: YOLOV1ObjectDetector,
    dataset: TensorflowObjectDetectionData,
    bbox_postprocessing: BoundingBoxesYOLOV1PostProcessing,
    dataset_subset: str,
    coco_json_filepath: str | Path,
    batch_size: int,
) -> Iterator[COCOResult]:
    data: Dataset = getattr(dataset, f"{dataset_subset}_dataset")
    data_filenames: list[str] = getattr(dataset, f"{dataset_subset}_images_filepaths")
    coco_dataset: COCODataset = read_coco_json(coco_json_filepath)
    file_name_to_coco_id_mapper: dict[
        str, int
    ] = create_image_file_name_to_coco_id_mapper(coco_dataset)
    file_name_to_wh_mapper: dict[
        str, dict[str, float]
    ] = create_image_file_name_to_wh_mapper(coco_dataset)

    data_filename_id: int = 0
    for image, _ in data:
        pred_bbox, pred_conf, pred_labels = model(image)
        bboxes, scores, labels, _ = bbox_postprocessing.postprocess(
            pred_bbox, pred_conf, pred_labels
        )

        for image_id in range(batch_size):
            data_filename: str = str(data_filenames[data_filename_id])
            image_wh: dict[str, float] = file_name_to_wh_mapper[data_filename]
            coco_results: list[COCOResult] = to_coco_results_format(
                bboxes=bboxes[image_id].numpy(),
                scores=scores[image_id].numpy(),
                labels=labels[image_id].numpy(),
                image_wh=image_wh,
                file_name_to_coco_id_mapper=file_name_to_coco_id_mapper,
                data_filename=data_filename,
            )

            for coco_result in coco_results:
                yield coco_result

            data_filename_id += 1


def _draw_bounding_box_on_images(
    images_dir: str | Path,
    coco_json_filepath: str | Path,
    coco_results_filepath: str | Path,
) -> Iterator[dict[str, npt.NDArray]]:
    images_dir = Path(images_dir)
    coco_dataset: COCODataset = read_coco_json(coco_json_filepath)
    coco_results: list[COCOResult] = read_coco_results_json(coco_results_filepath)
    image_file_name_to_result_mapper = create_image_file_name_to_result_mapper(
        coco_dataset=coco_dataset, coco_results=coco_results
    )

    for filepath in images_dir.iterdir():
        if filepath.is_file():
            result = image_file_name_to_result_mapper[filepath.name]
            image = iio.imread(filepath)
            bboxes_on_image_array: npt.NDArray = add_bounding_boxes_to_image(
                image=image,
                bboxes=result["boxes"],
                scores=result["scores"],
                labels=result["labels"],
            )
            yield {filepath.name: bboxes_on_image_array}


def read_coco_json(filepath: str | Path) -> COCODataset:
    filepath = Path(filepath)

    with filepath.open("rt") as f:
        data = json.load(f)

    info: COCOInfo = COCOInfo(
        description=str(data["info"]["description"]),
        version=str(data["info"]["version"]),
        year=int(data["info"]["year"]),
    )
    images: list[COCOImage] = [
        COCOImage(
            id=int(x["id"]),
            width=float(x["width"]),
            height=float(x["height"]),
            file_name=str(x["file_name"]),
            date_captured=str(x["date_captured"]),
        )
        for x in data["images"]
    ]
    annotations: list[COCOAnnotation] = [
        COCOAnnotation(
            id=int(x["id"]),
            image_id=int(x["image_id"]),
            category_id=int(x["category_id"]),
            bbox=[float(coord) for coord in x["bbox"]],
            iscrowd=int(x["iscrowd"]),
            area=float(x["area"]),
        )
        for x in data["annotations"]
    ]
    categories: list[COCOCategory] = [
        COCOCategory(
            id=x["id"],
            name=x["name"],
        )
        for x in data["categories"]
    ]

    return COCODataset(
        info=info, images=images, annotations=annotations, categories=categories
    )


def read_coco_results_json(filepath: str | Path) -> list[COCOResult]:
    filepath = Path(filepath)

    with filepath.open("rt") as f:
        data = json.load(f)

    return [
        COCOResult(
            image_id=int(x["image_id"]),
            category_id=int(x["category_id"]),
            bbox=[float(coord) for coord in x["bbox"]],
            score=float(x["score"]),
        )
        for x in data
    ]


def create_coco_result(
    image_id: int,
    x: np.floating[T_coord] | float,
    y: np.floating[T_coord] | float,
    width: np.floating[T_coord] | float,
    height: np.floating[T_coord] | float,
    category_id: np.floating[T_id] | float | int,
    score: np.floating[T_score] | float,
    image_wh: dict[str, float],
) -> COCOResult:
    image_width: float = float(image_wh["width"])
    image_height: float = float(image_wh["height"])

    return COCOResult(
        image_id=int(image_id),
        category_id=int(category_id),
        bbox=[
            round(float(x) * image_width, 1),
            round(float(y) * image_height, 1),
            round(float(width) * image_width, 1),
            round(float(height) * image_height, 1),
        ],
        score=round(float(score), 6),
    )


def create_image_file_name_to_coco_id_mapper(
    coco_dataset: COCODataset,
) -> dict[str, int]:
    file_name_to_coco_id_mapper: dict[str, int] = {}
    coco_images: list[COCOImage] = coco_dataset["images"]

    for coco_image in coco_images:
        file_name_to_coco_id_mapper[coco_image["file_name"]] = coco_image["id"]

    return file_name_to_coco_id_mapper


def create_image_file_name_to_wh_mapper(coco_dataset: COCODataset) -> dict[str, int]:
    file_name_to_wh_mapper: dict[str, dict[str, float]] = {}
    coco_images: list[COCOImage] = coco_dataset["images"]

    for coco_image in coco_images:
        file_name_to_wh_mapper[coco_image["file_name"]] = {
            "width": coco_image["width"],
            "height": coco_image["height"],
        }

    return file_name_to_wh_mapper


def create_image_file_name_to_result_mapper(
    coco_dataset: COCODataset, coco_results: list[COCOResult]
) -> dict[str, dict[str, list[list[float]] | list[str] | list[float]]]:
    coco_images: list[COCOImage] = coco_dataset["images"]
    label_mapper: dict[int, str] = create_label_mapper(coco_dataset)

    image_id_to_result_mapper: dict[
        int, dict[str, list[list[float]] | list[str] | list[float]]
    ] = {}
    file_name_to_result_mapper: dict[
        str, dict[str, list[list[float]] | list[str] | list[float]]
    ] = {}

    for coco_result in coco_results:
        image_id: int = coco_result["image_id"]
        image_id_to_result_mapper[image_id] = {"boxes": [], "labels": [], "scores": []}

        bbox: list[float] = coco_result["bbox"]
        label_id: int = coco_result["category_id"]
        label: str = label_mapper.get(label_id, str(label_id))
        score: float = coco_result["score"]

        image_id_to_result_mapper[image_id]["boxes"].append(bbox)
        image_id_to_result_mapper[image_id]["labels"].append(label)
        image_id_to_result_mapper[image_id]["scores"].append(score)

    for coco_image in coco_images:
        coco_image_id: int = coco_image["id"]
        file_name: str = coco_image["file_name"]
        file_name_to_result_mapper[file_name] = {
            "boxes": image_id_to_result_mapper[coco_image_id]["boxes"],
            "labels": image_id_to_result_mapper[coco_image_id]["labels"],
            "scores": image_id_to_result_mapper[coco_image_id]["scores"],
        }

    return file_name_to_result_mapper


def create_label_mapper(coco_dataset: COCODataset) -> dict[int, str]:
    label_mapper: dict[str, int] = {}
    coco_categories: list[COCOCategory] = coco_dataset["categories"]

    for coco_category in coco_categories:
        label_mapper[coco_category["id"]] = coco_category["name"]

    return label_mapper


def add_bounding_boxes_to_image(
    image: npt.NDArray,
    bboxes: list[float],
    scores: list[float],
    labels: list[str],
) -> npt.NDArray:
    image_width: int = int(image.shape[1])
    image_height: int = int(image.shape[0])
    num_channels: int = int(image.shape[2])

    bboxes_list: list[BoundingBox] = []

    for x, score, label in zip(bboxes, scores, labels):
        label = f"{label} ({score:.6f})"
        bboxes_list.append(
            BoundingBox(
                x1=x[0],
                y1=x[1],
                x2=x[0] + x[2],
                y2=x[1] + x[3],
                label=label,
            )
        )

    bboxes_on_image: BoundingBoxesOnImage = BoundingBoxesOnImage(
        bboxes_list, shape=(image_height, image_width, num_channels)
    )
    bboxes_on_image_array: npt.NDArray = (
        bboxes_on_image.clip_out_of_image().draw_on_image(image=image.astype("uint8"))
    )

    return bboxes_on_image_array


def to_coco_results_format(
    bboxes: npt.NDArray,
    scores: npt.NDArray,
    labels: npt.NDArray,
    image_wh: dict[str, float],
    file_name_to_coco_id_mapper: dict[str, int],
    data_filename: str | Path,
) -> list[COCOResult]:
    data_filename = Path(data_filename)
    coco_image_id: int = file_name_to_coco_id_mapper[data_filename.name]

    coco_results: list[COCOResult] = []

    for x, score, label_id in zip(bboxes, scores, labels):
        if not score < 1e-6:
            coco_results.append(
                create_coco_result(
                    image_id=coco_image_id,
                    x=x[1],
                    y=x[0],
                    width=(x[3] - x[1]),
                    height=(x[2] - x[0]),
                    category_id=label_id,
                    score=score,
                    image_wh=image_wh,
                )
            )

    return coco_results
