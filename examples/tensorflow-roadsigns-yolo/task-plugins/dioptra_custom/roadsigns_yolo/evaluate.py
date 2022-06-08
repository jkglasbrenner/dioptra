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

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dioptra import pyplugs


@pyplugs.register
def coco_evaluate(
    coco_json_filepath: str | Path,
    coco_results_filepath: str | Path,
    output_filepath: str | Path,
) -> None:
    coco_json_filepath = str(coco_json_filepath)
    coco_results_filepath = str(coco_results_filepath)
    output_filepath = Path(output_filepath)

    coco_gt = COCO(coco_json_filepath)
    coco_dt = coco_gt.loadRes(coco_results_filepath)
    coco_annotation_type = "bbox"

    coco_eval = COCOeval(coco_gt, coco_dt, coco_annotation_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    coco_eval_encoded: dict[str, float] = {
        "AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]": float(coco_eval.stats[0]),
        "AP @[ IoU=0.50      | area=   all | maxDets=100 ]": float(coco_eval.stats[1]),
        "AP @[ IoU=0.75      | area=   all | maxDets=100 ]": float(coco_eval.stats[2]),
        "AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]": float(coco_eval.stats[3]),
        "AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]": float(coco_eval.stats[4]),
        "AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]": float(coco_eval.stats[5]),
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]": float(coco_eval.stats[6]),
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]": float(coco_eval.stats[7]),
        "AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]": float(coco_eval.stats[8]),
        "AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]": float(coco_eval.stats[9]),
        "AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]": float(coco_eval.stats[10]),
        "AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]": float(coco_eval.stats[11]),
    }

    with output_filepath.open("wt") as f:
        json.dump(obj=coco_eval_encoded, fp=f)
