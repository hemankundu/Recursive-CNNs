''' Document Localization using Recursive CNN
 Maintainer : Khurram Javed
 Email : kjaved@ualberta.ca '''

import cv2
import numpy as np

import evaluation


def args_processor():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--imagePath", default="../058.jpg", help="Path to the document image")
    parser.add_argument("-o", "--outputPath", default="../output.jpg", help="Path to store the result")
    parser.add_argument("-cm", "--cornerModel", help="Model for corner point refinement",
                        default="../cornerModelWell")
    parser.add_argument("-cm_type", "--cornerModelType", help="Model type for corner point refinement",
                        default="reset")
    parser.add_argument("-dm", "--documentModel", help="Model for document corners detection",
                        default="../documentModelWell")
    parser.add_argument("-dm_type", "--documentModelType", help="Model type for document corners detection",
                        default="resnet")
    return parser.parse_args()


if __name__ == "__main__":
    args = args_processor()

    corners_extractor = evaluation.corner_extractor.GetCorners(args.documentModel, args.documentModelType)
    corner_refiner = evaluation.corner_refiner.corner_finder(args.cornerModel, args.cornerModelType)

    img = cv2.imread(args.imagePath)

    oImg = img

    extracted_corners = corners_extractor.get(oImg)
    
    x_coords, y_coords = extracted_corners[-2:]
    extracted_corners = extracted_corners[:-2]

    # print(extracted_corners)

    for x, y in zip(x_coords, y_coords):
        oImg = cv2.circle(oImg, (x, y), 5, (255,0,0), -1)

    corner_address = []
    # Refine the detected corners using corner refiner
    image_name = 0
    for corner in extracted_corners:
        image_name += 1
        corner_img = corner[1]
        refined_corner = np.array(corner_refiner.get_location(corner_img, args.retainFactor))

        # Converting from local co-ordinate to global co-ordinates of the image
        refined_corner[0] = int(refined_corner[0]*(corner[0][3] - corner[0][2]) + corner[0][2])
        refined_corner[1] = int(refined_corner[1]*(corner[0][1] - corner[0][0]) + corner[0][0])

        # Final results
        corner_address.append(refined_corner)

    for a in range(0, len(extracted_corners)):
        cv2.line(oImg, tuple(corner_address[a % 4]), tuple(corner_address[(a + 1) % 4]), (255, 0, 0), 4)

    cv2.imwrite(args.outputPath, oImg)
