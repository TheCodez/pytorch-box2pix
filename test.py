from argparse import ArgumentParser

import cv2

from prediction.predictor import Predictor


def run(file_path, show_boxes):
    cap = cv2.VideoCapture(file_path)
    detector = Predictor(show_boxes)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ssd, segmentation, instance = detector.run(frame)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser('Box2Pix with PyTorch')
    parser.add_argument('input', help='input video file')
    parser.add_argument('--show-boxes', action='store_true',
                        help='whether or not to also display boxes in the result')

    args = parser.parse_args()

    if args.input is not None:
        run(args.input, args.show_boxes)
