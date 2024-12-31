from typing import List, Tuple, Optional, Union
import cv2
import numpy as np

def get_recommended_box_thickness(x1: int, y1: int, x2: int, y2: int) -> int:
    """Calculate recommended box thickness based on box size."""
    return max(1, min(3, int(round((x2 - x1 + y2 - y1) / 400))))

def draw_bbox(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    title: str = "",
    box_thickness: Optional[int] = None,
) -> np.ndarray:
    """Draw a bounding box with optional title on an image."""
    if box_thickness is None:
        box_thickness = get_recommended_box_thickness(x1, y1, x2, y2)

    # Draw the bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=box_thickness, lineType=cv2.LINE_AA)

    if title:
        # Calculate text size and position
        font_scale = max(0.5, box_thickness * 0.3)
        (text_width, text_height), baseline = cv2.getTextSize(
            title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness=box_thickness
        )

        # Draw title background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width + 5, y1),
            color,
            thickness=-1,
        )

        # Draw title text
        cv2.putText(
            image,
            title,
            (x1 + 3, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness=box_thickness,
            lineType=cv2.LINE_AA,
        )

    return image

class ObjectDetectionVisualization:
    @classmethod
    def draw_detections(
        cls,
        *,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        box_thickness: Optional[int] = None,
        score_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Draw multiple detections on an image.
        
        Args:
            image: Image on which to draw the detections
            boxes: Predicted boxes in XYXY format with shape [N, 4]
            scores: Optional confidence scores with shape [N]
            labels: Optional list of label strings for each detection
            colors: Optional list of RGB colors for each class
            box_thickness: Optional thickness for bounding boxes
            score_threshold: Minimum confidence score to display a detection
            
        Returns:
            Image with drawn detections
        """
        if len(boxes) == 0:
            return image.copy()

        if scores is not None and len(scores) != len(boxes):
            raise ValueError("scores and boxes must have the same length")
        if labels is not None and len(labels) != len(boxes):
            raise ValueError("labels and boxes must have the same length")

        # Default colors if none provided
        if colors is None:
            colors = [(255, 0, 0)] * len(boxes)  # Default to red boxes
        elif len(colors) < len(boxes):
            colors = colors * (len(boxes) // len(colors) + 1)  # Repeat colors if needed

        # Sort by confidence score if available
        if scores is not None:
            order = np.argsort(scores)
            boxes = boxes[order]
            scores = scores[order]
            if labels is not None:
                labels = [labels[i] for i in order]
            colors = [colors[i] for i in order]

        result_image = image.copy()
        
        for i, box in enumerate(boxes):
            if scores is not None and scores[i] < score_threshold:
                continue

            x1, y1, x2, y2 = map(int, box)
            
            # Prepare title text
            title = ""
            if labels is not None:
                title += f"{labels[i]} "
            if scores is not None:
                title += f"{scores[i]:.2f}"

            current_box_thickness = box_thickness or get_recommended_box_thickness(x1, y1, x2, y2)

            result_image = draw_bbox(
                image=result_image,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                color=colors[i],
                title=title.strip(),
                box_thickness=current_box_thickness,
            )

        return result_image
